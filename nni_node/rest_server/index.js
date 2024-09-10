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
exports.UnitTestHelpers = exports.RestServer = void 0;
const strict_1 = __importDefault(require("assert/strict"));
const path_1 = __importDefault(require("path"));
const express_1 = __importStar(require("express"));
const express_ws_1 = __importDefault(require("express-ws"));
const http_proxy_1 = __importDefault(require("http-proxy"));
const ts_deferred_1 = require("ts-deferred");
const globals_1 = __importDefault(require("common/globals"));
const log_1 = require("common/log");
const tunerCommandChannel = __importStar(require("core/tuner_command_channel"));
const restHandler_1 = require("./restHandler");
const logger = log_1.getLogger('RestServer');
class RestServer {
    port;
    urlPrefix;
    server = null;
    constructor(port, urlPrefix) {
        strict_1.default(!urlPrefix.startsWith('/') && !urlPrefix.endsWith('/'));
        this.port = port;
        this.urlPrefix = urlPrefix;
        globals_1.default.shutdown.register('RestServer', this.shutdown.bind(this));
    }
    start() {
        logger.info(`Starting REST server at port ${this.port}, URL prefix: "/${this.urlPrefix}"`);
        const app = express_1.default();
        express_ws_1.default(app, undefined, { wsOptions: { maxPayload: 4 * 1024 * 1024 * 1024 } });
        app.use('/' + this.urlPrefix, rootRouter());
        app.all('*', (_req, res) => { res.status(404).send(`Outside prefix "/${this.urlPrefix}"`); });
        this.server = app.listen(this.port);
        const deferred = new ts_deferred_1.Deferred();
        this.server.on('listening', () => {
            if (this.port === 0) {
                this.port = this.server.address().port;
            }
            logger.info('REST server started.');
            deferred.resolve();
        });
        this.server.on('error', (error) => { globals_1.default.shutdown.criticalError('RestServer', error); });
        return deferred.promise;
    }
    shutdown() {
        logger.info('Stopping REST server.');
        if (this.server === null) {
            logger.warning('REST server is not running.');
            return Promise.resolve();
        }
        const deferred = new ts_deferred_1.Deferred();
        this.server.close(() => {
            logger.info('REST server stopped.');
            deferred.resolve();
        });
        return deferred.promise;
    }
}
exports.RestServer = RestServer;
function rootRouter() {
    const router = express_1.Router();
    router.use(express_1.default.json({ limit: '50mb' }));
    router.use('/api/v1/nni', restHandlerFactory());
    router.ws('/tuner', (ws, _req, _next) => { tunerCommandChannel.serveWebSocket(ws); });
    const logRouter = express_1.Router();
    logRouter.get('*', express_1.default.static(globals_1.default.paths.logDirectory));
    router.use('/logs', logRouter);
    router.use('/netron', netronProxy());
    router.get('*', express_1.default.static(webuiPath));
    router.get('*', (_req, res) => { res.sendFile(path_1.default.join(webuiPath, 'index.html')); });
    router.all('*', (_req, res) => { res.status(404).send('Not Found'); });
    return router;
}
function netronProxy() {
    const router = express_1.Router();
    const proxy = http_proxy_1.default.createProxyServer();
    router.all('*', (req, res) => {
        delete req.headers.host;
        proxy.web(req, res, { changeOrigin: true, target: netronUrl });
    });
    return router;
}
let webuiPath = path_1.default.resolve('static');
let netronUrl = 'https://netron.app';
let restHandlerFactory = restHandler_1.createRestHandler;
var UnitTestHelpers;
(function (UnitTestHelpers) {
    function getPort(server) {
        return server.port;
    }
    UnitTestHelpers.getPort = getPort;
    function setWebuiPath(mockPath) {
        webuiPath = path_1.default.resolve(mockPath);
    }
    UnitTestHelpers.setWebuiPath = setWebuiPath;
    function setNetronUrl(mockUrl) {
        netronUrl = mockUrl;
    }
    UnitTestHelpers.setNetronUrl = setNetronUrl;
    function disableNniManager() {
        restHandlerFactory = () => express_1.Router();
    }
    UnitTestHelpers.disableNniManager = disableNniManager;
    function reset() {
        webuiPath = path_1.default.resolve('static');
        netronUrl = 'https://netron.app';
        restHandlerFactory = restHandler_1.createRestHandler;
    }
    UnitTestHelpers.reset = reset;
})(UnitTestHelpers = exports.UnitTestHelpers || (exports.UnitTestHelpers = {}));
