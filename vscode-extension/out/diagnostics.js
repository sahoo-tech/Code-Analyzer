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
exports.DiagnosticsProvider = void 0;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
/**
 * Provides VS Code diagnostics (inline warnings/errors) from analysis results
 */
class DiagnosticsProvider {
    diagnosticCollection;
    constructor() {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('codeAnalyzer');
    }
    /**
     * Update diagnostics for a file based on analysis results
     */
    updateDiagnostics(filePath, analysisResult) {
        const uri = vscode.Uri.file(filePath);
        const diagnostics = [];
        // Convert vulnerabilities to diagnostics
        if (analysisResult.vulnerabilities) {
            for (const vuln of analysisResult.vulnerabilities) {
                if (this.matchesFile(vuln.file_path, filePath)) {
                    const diagnostic = this.createDiagnostic(vuln.line_number, `🔒 ${vuln.message}`, this.getSeverity(vuln.severity), 'security');
                    if (vuln.recommendation) {
                        diagnostic.message += `\n💡 ${vuln.recommendation}`;
                    }
                    diagnostics.push(diagnostic);
                }
            }
        }
        // Convert code smells to diagnostics
        if (analysisResult.code_smells) {
            for (const smell of analysisResult.code_smells) {
                if (this.matchesFile(smell.file_path, filePath)) {
                    const diagnostic = this.createDiagnostic(smell.line_number, `🔍 ${smell.message}`, this.getSeverity(smell.severity), 'code-smell');
                    diagnostics.push(diagnostic);
                }
            }
        }
        // Convert anti-patterns to diagnostics
        if (analysisResult.anti_patterns) {
            for (const pattern of analysisResult.anti_patterns) {
                if (this.matchesFile(pattern.file_path, filePath)) {
                    const diagnostic = this.createDiagnostic(pattern.line_number || 1, `⚠️ Anti-pattern: ${pattern.pattern_type} - ${pattern.description}`, vscode.DiagnosticSeverity.Warning, 'anti-pattern');
                    if (pattern.suggestion) {
                        diagnostic.message += `\n💡 ${pattern.suggestion}`;
                    }
                    diagnostics.push(diagnostic);
                }
            }
        }
        this.diagnosticCollection.set(uri, diagnostics);
    }
    /**
     * Clear all diagnostics
     */
    clear() {
        this.diagnosticCollection.clear();
    }
    /**
     * Clear diagnostics for a specific file
     */
    clearFile(filePath) {
        const uri = vscode.Uri.file(filePath);
        this.diagnosticCollection.delete(uri);
    }
    createDiagnostic(lineNumber, message, severity, code) {
        const line = Math.max(0, lineNumber - 1); // Convert to 0-indexed
        const range = new vscode.Range(line, 0, line, 100);
        const diagnostic = new vscode.Diagnostic(range, message, severity);
        diagnostic.source = 'Code Analyzer';
        diagnostic.code = code;
        return diagnostic;
    }
    getSeverity(severityString) {
        switch (severityString?.toLowerCase()) {
            case 'critical':
            case 'high':
                return vscode.DiagnosticSeverity.Error;
            case 'medium':
            case 'warning':
                return vscode.DiagnosticSeverity.Warning;
            case 'low':
            case 'info':
                return vscode.DiagnosticSeverity.Information;
            default:
                return vscode.DiagnosticSeverity.Hint;
        }
    }
    matchesFile(resultPath, targetPath) {
        if (!resultPath || !targetPath) {
            return true; // If no path specified, apply to target
        }
        const resultBasename = path.basename(resultPath);
        const targetBasename = path.basename(targetPath);
        return resultBasename === targetBasename || resultPath === targetPath;
    }
    dispose() {
        this.diagnosticCollection.dispose();
    }
}
exports.DiagnosticsProvider = DiagnosticsProvider;
//# sourceMappingURL=diagnostics.js.map