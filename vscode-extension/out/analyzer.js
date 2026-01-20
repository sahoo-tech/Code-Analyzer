"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.runAnalyzer = runAnalyzer;
exports.analyzeFile = analyzeFile;
exports.analyzeDirectory = analyzeDirectory;
exports.securityScan = securityScan;
exports.quickMetrics = quickMetrics;
exports.ragQuery = ragQuery;
const vscode = __importStar(require("vscode"));
const cp = __importStar(require("child_process"));
/**
 * Get the Python path from configuration
 */
function getPythonPath() {
    const config = vscode.workspace.getConfiguration('codeAnalyzer');
    return config.get('pythonPath') || 'python';
}
/**
 * Run the Python analyzer CLI and return JSON result
 */
async function runAnalyzer(args) {
    const pythonPath = getPythonPath();
    const command = ['-m', 'analyzer.cli', ...args, '--format', 'json'];
    return new Promise((resolve) => {
        const process = cp.spawn(pythonPath, command, {
            cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
            shell: true
        });
        let stdout = '';
        let stderr = '';
        process.stdout.on('data', (data) => {
            stdout += data.toString();
        });
        process.stderr.on('data', (data) => {
            stderr += data.toString();
        });
        process.on('close', (code) => {
            if (code === 0) {
                try {
                    // Find JSON in output (skip any log lines)
                    const jsonMatch = stdout.match(/\{[\s\S]*\}/);
                    if (jsonMatch) {
                        const data = JSON.parse(jsonMatch[0]);
                        resolve({ success: true, data });
                    }
                    else {
                        resolve({ success: true, data: { raw: stdout } });
                    }
                }
                catch (e) {
                    resolve({ success: true, data: { raw: stdout } });
                }
            }
            else {
                resolve({
                    success: false,
                    error: stderr || `Process exited with code ${code}`
                });
            }
        });
        process.on('error', (err) => {
            resolve({
                success: false,
                error: `Failed to run analyzer: ${err.message}`
            });
        });
    });
}
/**
 * Analyze a single file
 */
async function analyzeFile(filePath) {
    return runAnalyzer(['analyze', filePath]);
}
/**
 * Analyze a directory
 */
async function analyzeDirectory(dirPath) {
    return runAnalyzer(['analyze', dirPath]);
}
/**
 * Run security scan
 */
async function securityScan(targetPath) {
    return runAnalyzer(['security', targetPath]);
}
/**
 * Get quick metrics for a file
 */
async function quickMetrics(filePath) {
    return runAnalyzer(['metrics', filePath]);
}
/**
 * Query code with RAG AI
 */
async function ragQuery(question, projectPath) {
    return runAnalyzer(['rag', 'query', question, '--path', projectPath]);
}
//# sourceMappingURL=analyzer.js.map