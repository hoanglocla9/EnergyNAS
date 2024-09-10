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
        this.channel.onCommand((command) => {
            listener(command.slice(0, 2), command.slice(2));
        });
    }
    onError(listener) {
        this.channel.onError(listener);
    }
}
