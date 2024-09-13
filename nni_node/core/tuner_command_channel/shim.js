"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createDispatcherInterface = void 0;
const websocket_channel_1 = require("./websocket_channel");
async function createDispatcherInterface() {
    return new WsIpcInterface();
}
exports.createDispatcherInterface = createDispatcherInterface;
class WsIpcInterface {
    channel = websocket_channel_1.getWebSocketChannel();
    commandListener;
    errorListener;
    constructor() {
        this.channel.onCommand((command) => {
            const commandType = command.slice(0, 2);
            const content = command.slice(2);
            if (commandType === 'ER') {
                if (this.errorListener !== undefined) {
                    this.errorListener(new Error(content));
                }
            }
            else {
                if (this.commandListener !== undefined) {
                    this.commandListener(commandType, content);
                }
            }
        });
    }
    async init() {
        await this.channel.init();
    }
    sendCommand(commandType, content = '') {
        if (commandType !== 'PI') {
            this.channel.sendCommand(commandType + content);
            if (commandType === 'TE') {
                this.channel.shutdown();
            }
        }
    }
    onCommand(listener) {
        this.commandListener = listener;
    }
    onError(listener) {
        this.errorListener = listener;
    }
}
