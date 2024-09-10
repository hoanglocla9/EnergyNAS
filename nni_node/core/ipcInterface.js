"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnitTestHelpers = exports.encodeCommand = exports.createDispatcherInterface = void 0;
const shim = __importStar(require("./tuner_command_channel/shim"));
let tunerDisabled = false;
async function createDispatcherInterface() {
    if (!tunerDisabled) {
        return await shim.createDispatcherInterface();
    }
    else {
        return new DummyIpcInterface();
    }
}
exports.createDispatcherInterface = createDispatcherInterface;
function encodeCommand(commandType, content) {
    const contentBuffer = Buffer.from(content);
    const contentLengthBuffer = Buffer.from(contentBuffer.length.toString().padStart(14, '0'));
    return Buffer.concat([Buffer.from(commandType), contentLengthBuffer, contentBuffer]);
}
exports.encodeCommand = encodeCommand;
class DummyIpcInterface {
    async init() { }
    sendCommand(_commandType, _content) { }
    onCommand(_listener) { }
    onError(_listener) { }
}
var UnitTestHelpers;
(function (UnitTestHelpers) {
    function disableTuner() {
        tunerDisabled = true;
    }
    UnitTestHelpers.disableTuner = disableTuner;
})(UnitTestHelpers = exports.UnitTestHelpers || (exports.UnitTestHelpers = {}));
