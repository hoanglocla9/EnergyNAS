"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.initLogStream = void 0;
const fs_1 = __importDefault(require("fs"));
const promises_1 = require("timers/promises");
const util_1 = __importDefault(require("util"));
const writePromise = util_1.default.promisify(fs_1.default.write);
class LogStreamImpl {
    buffer = [];
    flushing = false;
    logFileFd;
    toConsole;
    constructor(logFile, toConsole) {
        this.logFileFd = fs_1.default.openSync(logFile, 'a');
        this.toConsole = toConsole;
    }
    writeLine(line) {
        this.buffer.push(line);
        this.flush();
    }
    writeLineSync(line) {
        if (this.toConsole) {
            console.log(line);
        }
        fs_1.default.writeSync(this.logFileFd, line + '\n');
    }
    async close() {
        while (this.flushing) {
            await promises_1.setTimeout();
        }
        fs_1.default.closeSync(this.logFileFd);
        this.logFileFd = 2;
        this.toConsole = false;
    }
    async flush() {
        if (this.flushing) {
            return;
        }
        this.flushing = true;
        while (this.buffer.length > 0) {
            const lines = this.buffer.join('\n');
            this.buffer.length = 0;
            if (this.toConsole) {
                console.log(lines);
            }
            await writePromise(this.logFileFd, lines + '\n');
        }
        this.flushing = false;
    }
}
function initLogStream(args, paths) {
    return new LogStreamImpl(paths.nniManagerLog, args.foreground);
}
exports.initLogStream = initLogStream;
