"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createPaths = void 0;
const strict_1 = __importDefault(require("assert/strict"));
const fs_1 = __importDefault(require("fs"));
const os_1 = __importDefault(require("os"));
const path_1 = __importDefault(require("path"));
function createPaths(args) {
    strict_1.default(path_1.default.isAbsolute(args.experimentsDirectory), `Command line arg --experiments-directory "${args.experimentsDirectory}" is not absoulte`);
    const experimentRoot = path_1.default.join(args.experimentsDirectory, args.experimentId);
    const logDirectory = path_1.default.join(experimentRoot, 'log');
    fs_1.default.mkdirSync(logDirectory, { recursive: true });
    const nniManagerLog = path_1.default.join(logDirectory, 'nnimanager.log');
    const experimentsList = path_1.default.join(os_1.default.homedir(), 'nni-experiments', '.experiment');
    return {
        experimentRoot,
        experimentsDirectory: args.experimentsDirectory,
        experimentsList,
        logDirectory,
        nniManagerLog,
    };
}
exports.createPaths = createPaths;
