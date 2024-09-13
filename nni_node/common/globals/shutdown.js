"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnitTestHelpers = exports.ShutdownManager = void 0;
const log_1 = require("common/log");
const logger = log_1.getRobustLogger('ShutdownManager');
class ShutdownManager {
    processStatus = 'initializing';
    modules = new Map();
    hasError = false;
    register(moduleName, shutdownCallback) {
        if (this.modules.has(moduleName)) {
            logger.error(`Module ${moduleName} has registered twice.`, new Error().stack);
        }
        this.modules.set(moduleName, shutdownCallback);
    }
    initiate(reason) {
        if (this.processStatus === 'stopping') {
            logger.warning('initiate() invoked but already stopping:', reason);
        }
        else {
            logger.info('Initiate shutdown:', reason);
            this.shutdown();
        }
    }
    criticalError(moduleName, error) {
        logger.critical(`Critical error ocurred in module ${moduleName}:`, error);
        this.hasError = true;
        if (this.processStatus === 'initializing') {
            logger.error('Starting failed.');
            process.exit(1);
        }
        else if (this.processStatus !== 'stopping') {
            this.modules.delete(moduleName);
            this.shutdown();
        }
    }
    notifyInitializeComplete() {
        if (this.processStatus === 'initializing') {
            this.processStatus = 'running';
        }
        else {
            logger.error('notifyInitializeComplete() invoked in status', this.processStatus);
        }
    }
    shutdown() {
        this.processStatus = 'stopping';
        const promises = Array.from(this.modules).map(async ([moduleName, callback]) => {
            try {
                await callback();
            }
            catch (error) {
                logger.error(`Error during shutting down ${moduleName}:`, error);
                this.hasError = true;
            }
            this.modules.delete(moduleName);
        });
        const timeoutTimer = setTimeout(async () => {
            try {
                logger.error('Following modules failed to shut down in time:', this.modules.keys());
                await global.nni.logStream.close();
            }
            finally {
                process.exit(1);
            }
        }, shutdownTimeout);
        Promise.all(promises).then(async () => {
            try {
                clearTimeout(timeoutTimer);
                logger.info('Shutdown complete.');
                await global.nni.logStream.close();
            }
            finally {
                process.exit(this.hasError ? 1 : 0);
            }
        });
    }
}
exports.ShutdownManager = ShutdownManager;
let shutdownTimeout = 60_000;
var UnitTestHelpers;
(function (UnitTestHelpers) {
    function setShutdownTimeout(ms) {
        shutdownTimeout = ms;
    }
    UnitTestHelpers.setShutdownTimeout = setShutdownTimeout;
})(UnitTestHelpers = exports.UnitTestHelpers || (exports.UnitTestHelpers = {}));
