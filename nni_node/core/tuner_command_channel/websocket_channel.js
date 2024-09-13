"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnitTestHelpers = exports.serveWebSocket = exports.getWebSocketChannel = void 0;
const strict_1 = __importDefault(require("assert/strict"));
const events_1 = require("events");
const deferred_1 = require("common/deferred");
const log_1 = require("common/log");
const logger = log_1.getLogger('tuner_command_channel.WebSocketChannel');
function getWebSocketChannel() {
    return channelSingleton;
}
exports.getWebSocketChannel = getWebSocketChannel;
function serveWebSocket(ws) {
    channelSingleton.serveWebSocket(ws);
}
exports.serveWebSocket = serveWebSocket;
class WebSocketChannelImpl {
    deferredInit = new deferred_1.Deferred();
    emitter = new events_1.EventEmitter();
    heartbeatTimer;
    serving = false;
    waitingPong = false;
    ws;
    serveWebSocket(ws) {
        if (this.ws === undefined) {
            logger.debug('Connected.');
        }
        else {
            logger.warning('Reconnecting. Drop previous connection.');
            this.dropConnection('Reconnected');
        }
        this.serving = true;
        this.ws = ws;
        this.ws.on('close', this.handleWsClose);
        this.ws.on('error', this.handleWsError);
        this.ws.on('message', this.handleWsMessage);
        this.ws.on('pong', this.handleWsPong);
        this.heartbeatTimer = setInterval(this.heartbeat.bind(this), heartbeatInterval);
        this.deferredInit.resolve();
    }
    init() {
        if (this.ws === undefined) {
            logger.debug('Waiting connection...');
            setTimeout(() => {
                if (!this.deferredInit.settled) {
                    const msg = 'Tuner did not connect in 10 seconds. Please check tuner (dispatcher) log.';
                    this.deferredInit.reject(new Error('tuner_command_channel: ' + msg));
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
    handleWsClose = () => {
        this.handleError(new Error('tuner_command_channel: Tuner closed connection'));
    };
    handleWsError = (error) => {
        this.handleError(error);
    };
    handleWsMessage = (data, _isBinary) => {
        this.receive(data);
    };
    handleWsPong = () => {
        this.waitingPong = false;
    };
    dropConnection(reason) {
        if (this.ws === undefined) {
            return;
        }
        this.serving = false;
        this.waitingPong = false;
        clearInterval(this.heartbeatTimer);
        this.ws.off('close', this.handleWsClose);
        this.ws.off('error', this.handleWsError);
        this.ws.off('message', this.handleWsMessage);
        this.ws.off('pong', this.handleWsPong);
        this.ws.on('close', () => {
            logger.info('Connection dropped');
        });
        this.ws.on('message', (data, _isBinary) => {
            logger.error('Received message after reconnect:', data);
        });
        this.ws.on('pong', () => {
            logger.error('Received pong after reconnect.');
        });
        this.ws.close(1001, reason);
    }
    heartbeat() {
        if (this.waitingPong) {
            this.ws.terminate();
            this.handleError(new Error('tuner_command_channel: Tuner loses responsive'));
        }
        this.waitingPong = true;
        this.ws.ping();
    }
    receive(data) {
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
