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
require("app-module-path/register");
const typescript_ioc_1 = require("typescript-ioc");
const component = __importStar(require("common/component"));
const datastore_1 = require("common/datastore");
const globals_1 = __importStar(require("common/globals"));
const log_1 = require("common/log");
const manager_1 = require("common/manager");
const tensorboardManager_1 = require("common/tensorboardManager");
const nniDataStore_1 = require("core/nniDataStore");
const nnimanager_1 = require("core/nnimanager");
const sqlDatabase_1 = require("core/sqlDatabase");
const experiments_manager_1 = require("extensions/experiments_manager");
const nniTensorboardManager_1 = require("extensions/nniTensorboardManager");
const rest_server_1 = require("rest_server");
const logger = log_1.getLogger('main');
async function start() {
    logger.info('Start NNI manager');
    typescript_ioc_1.Container.bind(manager_1.Manager).to(nnimanager_1.NNIManager).scope(typescript_ioc_1.Scope.Singleton);
    typescript_ioc_1.Container.bind(datastore_1.Database).to(sqlDatabase_1.SqlDB).scope(typescript_ioc_1.Scope.Singleton);
    typescript_ioc_1.Container.bind(datastore_1.DataStore).to(nniDataStore_1.NNIDataStore).scope(typescript_ioc_1.Scope.Singleton);
    typescript_ioc_1.Container.bind(tensorboardManager_1.TensorboardManager).to(nniTensorboardManager_1.NNITensorboardManager).scope(typescript_ioc_1.Scope.Singleton);
    const ds = component.get(datastore_1.DataStore);
    await ds.init();
    const restServer = new rest_server_1.RestServer(globals_1.default.args.port, globals_1.default.args.urlPrefix);
    await restServer.start();
    experiments_manager_1.initExperimentsManager();
    globals_1.default.shutdown.notifyInitializeComplete();
}
process.on('SIGTERM', () => { globals_1.default.shutdown.initiate('SIGTERM'); });
process.on('SIGBREAK', () => { globals_1.default.shutdown.initiate('SIGBREAK'); });
process.on('SIGINT', () => { globals_1.default.shutdown.initiate('SIGINT'); });
globals_1.initGlobals();
start().then(() => {
    logger.debug('start() returned.');
}).catch((error) => {
    try {
        logger.error('Failed to start:', error);
    }
    catch (loggerError) {
        console.error('Failed to start:', error);
        console.error('Seems logger is faulty:', loggerError);
    }
    process.exit(1);
});
