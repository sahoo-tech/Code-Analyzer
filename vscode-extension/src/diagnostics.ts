import * as vscode from 'vscode';
import * as path from 'path';

/**
 * Provides VS Code diagnostics (inline warnings/errors) from analysis results
 */
export class DiagnosticsProvider implements vscode.Disposable {
    private diagnosticCollection: vscode.DiagnosticCollection;

    constructor() {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('codeAnalyzer');
    }

    /**
     * Update diagnostics for a file based on analysis results
     */
    updateDiagnostics(filePath: string, analysisResult: any): void {
        const uri = vscode.Uri.file(filePath);
        const diagnostics: vscode.Diagnostic[] = [];

        // Convert vulnerabilities to diagnostics
        if (analysisResult.vulnerabilities) {
            for (const vuln of analysisResult.vulnerabilities) {
                if (this.matchesFile(vuln.file_path, filePath)) {
                    const diagnostic = this.createDiagnostic(
                        vuln.line_number,
                        `🔒 ${vuln.message}`,
                        this.getSeverity(vuln.severity),
                        'security'
                    );
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
                    const diagnostic = this.createDiagnostic(
                        smell.line_number,
                        `🔍 ${smell.message}`,
                        this.getSeverity(smell.severity),
                        'code-smell'
                    );
                    diagnostics.push(diagnostic);
                }
            }
        }

        // Convert anti-patterns to diagnostics
        if (analysisResult.anti_patterns) {
            for (const pattern of analysisResult.anti_patterns) {
                if (this.matchesFile(pattern.file_path, filePath)) {
                    const diagnostic = this.createDiagnostic(
                        pattern.line_number || 1,
                        `⚠️ Anti-pattern: ${pattern.pattern_type} - ${pattern.description}`,
                        vscode.DiagnosticSeverity.Warning,
                        'anti-pattern'
                    );
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
    clear(): void {
        this.diagnosticCollection.clear();
    }

    /**
     * Clear diagnostics for a specific file
     */
    clearFile(filePath: string): void {
        const uri = vscode.Uri.file(filePath);
        this.diagnosticCollection.delete(uri);
    }

    private createDiagnostic(
        lineNumber: number,
        message: string,
        severity: vscode.DiagnosticSeverity,
        code: string
    ): vscode.Diagnostic {
        const line = Math.max(0, lineNumber - 1); // Convert to 0-indexed
        const range = new vscode.Range(line, 0, line, 100);

        const diagnostic = new vscode.Diagnostic(range, message, severity);
        diagnostic.source = 'Code Analyzer';
        diagnostic.code = code;

        return diagnostic;
    }

    private getSeverity(severityString: string): vscode.DiagnosticSeverity {
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

    private matchesFile(resultPath: string, targetPath: string): boolean {
        if (!resultPath || !targetPath) {
            return true; // If no path specified, apply to target
        }
        const resultBasename = path.basename(resultPath);
        const targetBasename = path.basename(targetPath);
        return resultBasename === targetBasename || resultPath === targetPath;
    }

    dispose(): void {
        this.diagnosticCollection.dispose();
    }
}
