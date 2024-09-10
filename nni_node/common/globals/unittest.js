"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.resetGlobals = void 0;
const os_1 = __importDefault(require("os"));
const path_1 = __importDefault(require("path"));
const paths_1 = require("./paths");
function resetGlobals() {
    const args = {
        port: 8080,
        experimentId: 'unittest',
        action: 'create',
        experimentsDirectory: path_1.default.join(os_1.default.homedir(), 'nni-experiments'),
        logLevel: 'info',
        foreground: false,
        urlPrefix: '',
        tunerCommandChannel: null,
        pythonInterpreter: 'python',
        mode: 'unittest'
    };
    const paths = paths_1.createPaths(args);
    const logStream = {
        writeLine: (_line) => { },
        writeLineSync: (_line) => { },
        close: async () => { }
    };
    const shutdown = {
        register: (..._) => { },
    };
    const globalAsAny = global;
    const utGlobals = { args, paths, logStream, shutdown, reset: resetGlobals };
    if (globalAsAny.nni === undefined) {
        globalAsAny.nni = utGlobals;
    }
    else {
        Object.assign(globalAsAny.nni, utGlobals);
    }
}
exports.resetGlobals = resetGlobals;
function isUnitTest() {
    const event = process.env['npm_lifecycle_event'] ?? '';
    return event.startsWith('test') || event === 'mocha' || event === 'nyc';
}
if (isUnitTest()) {
    resetGlobals();
}
const globals = global.nni;
exports.default = globals;
