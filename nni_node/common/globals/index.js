"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.initGlobals = void 0;
const strict_1 = __importDefault(require("assert/strict"));
const arguments_1 = require("./arguments");
const log_stream_1 = require("./log_stream");
const paths_1 = require("./paths");
const shutdown_1 = require("./shutdown");
if (global.nni === undefined) {
    global.nni = {};
}
const globals = global.nni;
exports.default = globals;
function initGlobals() {
    strict_1.default.deepEqual(global.nni, {});
    const args = arguments_1.parseArgs(process.argv.slice(2));
    const paths = paths_1.createPaths(args);
    const logStream = log_stream_1.initLogStream(args, paths);
    const shutdown = new shutdown_1.ShutdownManager();
    const globals = { args, paths, logStream, shutdown };
    Object.assign(global.nni, globals);
}
exports.initGlobals = initGlobals;
