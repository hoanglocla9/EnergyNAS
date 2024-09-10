"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.runPythonScript = void 0;
const child_process_1 = require("child_process");
const globals_1 = __importDefault(require("./globals"));
const log_1 = require("./log");
const logger = log_1.getLogger('pythonScript');
async function runPythonScript(script, logTag) {
    const proc = child_process_1.spawn(globals_1.default.args.pythonInterpreter, ['-c', script]);
    let stdout = '';
    let stderr = '';
    proc.stdout.on('data', (data) => { stdout += data; });
    proc.stderr.on('data', (data) => { stderr += data; });
    const procPromise = new Promise((resolve, reject) => {
        proc.on('error', (err) => { reject(err); });
        proc.on('exit', () => { resolve(); });
    });
    await procPromise;
    if (stderr) {
        if (logTag) {
            logger.warning(`Python script [${logTag}] has stderr:`, stderr);
        }
        else {
            logger.warning('Python script has stderr.');
            logger.warning('  script:', script);
            logger.warning('  stderr:', stderr);
        }
    }
    return stdout;
}
exports.runPythonScript = runPythonScript;
