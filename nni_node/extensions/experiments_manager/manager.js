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
exports.ExperimentsManager = void 0;
const strict_1 = __importDefault(require("assert/strict"));
const fs_1 = __importDefault(require("fs"));
const timersPromises = __importStar(require("timers/promises"));
const ts_deferred_1 = require("ts-deferred");
const log_1 = require("common/log");
const globals_1 = __importDefault(require("common/globals"));
const utils_1 = require("common/utils");
const utils_2 = require("./utils");
const logger = log_1.getLogger('experiments_manager');
class ExperimentsManager {
    profileUpdateTimer = {};
    constructor() {
        globals_1.default.shutdown.register('experiments_manager', this.cleanUp.bind(this));
    }
    async getExperimentsInfo() {
        const fileInfo = await utils_2.withLock(globals_1.default.paths.experimentsList, () => this.readExperimentsInfo());
        const experimentsInformation = JSON.parse(fileInfo.buffer.toString());
        const expIdList = Object.keys(experimentsInformation).filter((expId) => {
            return experimentsInformation[expId]['status'] !== 'STOPPED';
        });
        const updateList = (await Promise.all(expIdList.map((expId) => {
            return this.checkCrashed(expId, experimentsInformation[expId]['pid']);
        }))).filter(crashedInfo => crashedInfo.isCrashed);
        if (updateList.length > 0) {
            const result = await utils_2.withLock(globals_1.default.paths.experimentsList, () => {
                return this.updateAllStatus(updateList.map(crashedInfo => crashedInfo.experimentId), fileInfo.mtime);
            });
            if (result !== undefined) {
                return JSON.parse(JSON.stringify(Object.keys(result).map(key => result[key])));
            }
            else {
                await timersPromises.setTimeout(500);
                return await this.getExperimentsInfo();
            }
        }
        else {
            return JSON.parse(JSON.stringify(Object.keys(experimentsInformation).map(key => experimentsInformation[key])));
        }
    }
    setExperimentInfo(experimentId, key, value) {
        try {
            if (this.profileUpdateTimer[key] !== undefined) {
                clearTimeout(this.profileUpdateTimer[key]);
                this.profileUpdateTimer[key] = undefined;
            }
            utils_2.withLockNoWait(globals_1.default.paths.experimentsList, () => {
                const experimentsInformation = JSON.parse(fs_1.default.readFileSync(globals_1.default.paths.experimentsList).toString());
                strict_1.default(experimentId in experimentsInformation, `Experiment Manager: Experiment Id ${experimentId} not found, this should not happen`);
                if (value !== undefined) {
                    experimentsInformation[experimentId][key] = value;
                }
                else {
                    delete experimentsInformation[experimentId][key];
                }
                fs_1.default.writeFileSync(globals_1.default.paths.experimentsList, JSON.stringify(experimentsInformation, null, 4));
            });
        }
        catch (err) {
            logger.error(err);
            logger.debug(`Experiment Manager: Retry set key value: ${experimentId} {${key}: ${value}}`);
            if (err.code === 'EEXIST' || err.message === 'File has been locked.') {
                this.profileUpdateTimer[key] = setTimeout(() => this.setExperimentInfo(experimentId, key, value), 100);
            }
        }
    }
    readExperimentsInfo() {
        const buffer = fs_1.default.readFileSync(globals_1.default.paths.experimentsList);
        const mtime = fs_1.default.statSync(globals_1.default.paths.experimentsList).mtimeMs;
        return { buffer: buffer, mtime: mtime };
    }
    async checkCrashed(expId, pid) {
        const alive = await utils_1.isAlive(pid);
        return { experimentId: expId, isCrashed: !alive };
    }
    updateAllStatus(updateList, timestamp) {
        if (timestamp !== fs_1.default.statSync(globals_1.default.paths.experimentsList).mtimeMs) {
            return;
        }
        else {
            const experimentsInformation = JSON.parse(fs_1.default.readFileSync(globals_1.default.paths.experimentsList).toString());
            updateList.forEach((expId) => {
                if (experimentsInformation[expId]) {
                    experimentsInformation[expId]['status'] = 'STOPPED';
                    delete experimentsInformation[expId]['port'];
                }
                else {
                    logger.error(`Experiment Manager: Experiment Id ${expId} not found, this should not happen`);
                }
            });
            fs_1.default.writeFileSync(globals_1.default.paths.experimentsList, JSON.stringify(experimentsInformation, null, 4));
            return experimentsInformation;
        }
    }
    async cleanUp() {
        const deferred = new ts_deferred_1.Deferred();
        if (this.isUndone()) {
            logger.debug('Experiment manager: something undone');
            setTimeout(((deferred) => {
                if (this.isUndone()) {
                    deferred.reject(new Error('Still has undone after 5s, forced stop.'));
                }
                else {
                    deferred.resolve();
                }
            }).bind(this), 5 * 1000, deferred);
        }
        else {
            logger.debug('Experiment manager: all clean up');
            deferred.resolve();
        }
        return deferred.promise;
    }
    isUndone() {
        return Object.keys(this.profileUpdateTimer).filter((key) => {
            return this.profileUpdateTimer[key] !== undefined;
        }).length > 0;
    }
}
exports.ExperimentsManager = ExperimentsManager;
