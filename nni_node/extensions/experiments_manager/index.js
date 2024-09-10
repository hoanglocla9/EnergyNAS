"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.UnitTestHelpers = exports.getExperimentsManager = exports.initExperimentsManager = exports.ExperimentsManager = void 0;
const manager_1 = require("./manager");
var manager_2 = require("./manager");
Object.defineProperty(exports, "ExperimentsManager", { enumerable: true, get: function () { return manager_2.ExperimentsManager; } });
let singleton = null;
function initExperimentsManager() {
    getExperimentsManager();
}
exports.initExperimentsManager = initExperimentsManager;
function getExperimentsManager() {
    if (singleton === null) {
        singleton = new manager_1.ExperimentsManager();
    }
    return singleton;
}
exports.getExperimentsManager = getExperimentsManager;
var UnitTestHelpers;
(function (UnitTestHelpers) {
    function setExperimentsManager(experimentsManager) {
        singleton = experimentsManager;
    }
    UnitTestHelpers.setExperimentsManager = setExperimentsManager;
})(UnitTestHelpers = exports.UnitTestHelpers || (exports.UnitTestHelpers = {}));
