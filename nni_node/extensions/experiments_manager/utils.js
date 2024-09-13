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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.withLock = exports.withLockNoWait = void 0;
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const timersPromises = __importStar(require("timers/promises"));
const glob_1 = __importDefault(require("glob"));
const lockfile_1 = __importDefault(require("lockfile"));
const lockStale = 2000;
const retry = 100;
function withLockNoWait(protectedFile, func) {
    const lockName = path_1.default.join(path_1.default.dirname(protectedFile), path_1.default.basename(protectedFile) + `.lock.${process.pid}`);
    const lockPath = path_1.default.join(path_1.default.dirname(protectedFile), path_1.default.basename(protectedFile) + '.lock.*');
    const lockFileNames = glob_1.default.sync(lockPath);
    const canLock = lockFileNames.map((fileName) => {
        return fs_1.default.existsSync(fileName) && Date.now() - fs_1.default.statSync(fileName).mtimeMs < lockStale;
    }).filter(unexpired => unexpired === true).length === 0;
    if (!canLock) {
        throw new Error('File has been locked.');
    }
    lockfile_1.default.lockSync(lockName, { stale: lockStale });
    const result = func();
    lockfile_1.default.unlockSync(lockName);
    return result;
}
exports.withLockNoWait = withLockNoWait;
async function withLock(protectedFile, func) {
    for (let i = 0; i < retry; i += 1) {
        try {
            return withLockNoWait(protectedFile, func);
        }
        catch (error) {
            if (error.code === 'EEXIST' || error.message === 'File has been locked.') {
                await timersPromises.setTimeout(50);
            }
            else {
                throw error;
            }
        }
    }
    throw new Error('Lock file out of retries.');
}
exports.withLock = withLock;
