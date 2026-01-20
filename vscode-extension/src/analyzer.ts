import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';

export interface AnalysisResult {
    success: boolean;
    data?: any;
    error?: string;
}

export interface Vulnerability {
    type: string;
    severity: string;
    message: string;
    file_path: string;
    line_number: number;
    recommendation?: string;
}

export interface CodeSmell {
    type: string;
    severity: string;
    message: string;
    file_path: string;
    line_number: number;
}

export interface MetricsResult {
    cyclomatic_complexity: number;
    cognitive_complexity: number;
    lines_of_code: number;
    maintainability_index: number;
}

/**
 * Get the Python path from configuration
 */
function getPythonPath(): string {
    const config = vscode.workspace.getConfiguration('codeAnalyzer');
    return config.get<string>('pythonPath') || 'python';
}

/**
 * Run the Python analyzer CLI and return JSON result
 */
export async function runAnalyzer(args: string[]): Promise<AnalysisResult> {
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
                    } else {
                        resolve({ success: true, data: { raw: stdout } });
                    }
                } catch (e) {
                    resolve({ success: true, data: { raw: stdout } });
                }
            } else {
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
export async function analyzeFile(filePath: string): Promise<AnalysisResult> {
    return runAnalyzer(['analyze', filePath]);
}

/**
 * Analyze a directory
 */
export async function analyzeDirectory(dirPath: string): Promise<AnalysisResult> {
    return runAnalyzer(['analyze', dirPath]);
}

/**
 * Run security scan
 */
export async function securityScan(targetPath: string): Promise<AnalysisResult> {
    return runAnalyzer(['security', targetPath]);
}

/**
 * Get quick metrics for a file
 */
export async function quickMetrics(filePath: string): Promise<AnalysisResult> {
    return runAnalyzer(['metrics', filePath]);
}

/**
 * Query code with RAG AI
 */
export async function ragQuery(question: string, projectPath: string): Promise<AnalysisResult> {
    return runAnalyzer(['rag', 'query', question, '--path', projectPath]);
}
