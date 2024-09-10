"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnitTestHelpers = exports.serveWebSocket = exports.getWebSocketChannel = void 0;
const strict_1 = __importDefault(require("assert/strict"));
const events_1 = require("events");
const ts_deferred_1 = require("ts-deferred");
const log_1 = require("common/log");
const logger = log_1.getLogger('tuner_command_channel.WebSocketChannel');
function getWebSocketChannel() {
    return channelSingleton;
}
exports.getWebSocketChannel = getWebSocketChannel;
function serveWebSocket(ws) {
    channelSingleton.setWebSocket(ws);
}
exports.serveWebSocket = serveWebSocket;
class WebSocketChannelImpl {
    deferredInit = new ts_deferred_1.Deferred();
    emitter = new events_1.EventEmitter();
    heartbeatTimer;
    serving = false;
    waitingPong = false;
    ws;
    setWebSocket(ws) {
        if (this.ws !== undefined) {
            logger.error('A second client is trying to connect.');
            ws.close(4030, 'Already serving a tuner');
            return;
        }
        if (this.deferredInit === null) {
            logger.error('Connection timed out.');
            ws.close(4080, 'Timeout');
            return;
        }
        logger.debug('Connected.');
        this.serving = true;
        this.ws = ws;
        ws.on('close', () => { this.handleError(new Error('tuner_command_channel: Tuner closed connection')); });
        ws.on('error', this.handleError.bind(this));
        ws.on('message', this.receive.bind(this));
        ws.on('pong', () => { this.waitingPong = false; });
        this.heartbeatTimer = setInterval(this.heartbeat.bind(this), heartbeatInterval);
        this.deferredInit.resolve();
        this.deferredInit = null;
    }
    init() {
        if (this.ws === undefined) {
            logger.debug('Waiting connection...');
            setTimeout(() => {
                if (this.deferredInit !== null) {
                    const msg = 'Tuner did not connect in 10 seconds. Please check tuner (dispatcher) log.';
                    this.deferredInit.reject(new Error('tuner_command_channel: ' + msg));
                    this.deferredInit = null;
                }
            }, 10000);
            return this.deferredInit.promise;
        }
        else {
            logger.debug('Initialized.');
            return Promise.resolve();
        }
    }
    async shutdown() {
        if (this.ws === undefined) {
            return;
        }
        clearInterval(this.heartbeatTimer);
        this.serving = false;
        this.emitter.removeAllListeners();
    }
    sendCommand(command) {
        strict_1.default.ok(this.ws !== undefined);
        logger.debug('Sending', command);
        this.ws.send(command);
        if (this.ws.bufferedAmount > command.length + 1000) {
            logger.warning('Sending too fast! Try to reduce the frequency of intermediate results.');
        }
    }
    onCommand(callback) {
        this.emitter.on('command', callback);
    }
    onError(callback) {
        this.emitter.on('error', callback);
    }
    heartbeat() {
        if (this.waitingPong) {
            this.ws.terminate();
            this.handleError(new Error('tuner_command_channel: Tuner loses responsive'));
        }
        this.waitingPong = true;
        this.ws.ping();
    }
    receive(data, _isBinary) {
        logger.debug('Received', data);
        this.emitter.emit('command', data.toString());
    }
    handleError(error) {
        if (!this.serving) {
            logger.debug('Silent error:', error);
            return;
        }
        logger.error('Error:', error);
        clearInterval(this.heartbeatTimer);
        this.emitter.emit('error', error);
        this.serving = false;
    }
}
const channelSingleton = new WebSocketChannelImpl();
let heartbeatInterval = 5000;
var UnitTestHelpers;
(function (UnitTestHelpers) {
    function setHeartbeatInterval(ms) {
        heartbeatInterval = ms;
    }
    UnitTestHelpers.setHeartbeatInterval = setHeartbeatInterval;
})(UnitTestHelpers = exports.UnitTestHelpers || (exports.UnitTestHelpers = {}));
