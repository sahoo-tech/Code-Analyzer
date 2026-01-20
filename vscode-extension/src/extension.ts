import * as vscode from 'vscode';
import * as path from 'path';
import * as analyzer from './analyzer';
import { DiagnosticsProvider } from './diagnostics';
import { ResultsPanel } from './views/resultsPanel';

let diagnosticsProvider: DiagnosticsProvider;
let outputChannel: vscode.OutputChannel;
let statusBarItem: vscode.StatusBarItem;

export function activate(context: vscode.ExtensionContext) {
    console.log('Code Analyzer extension is now active');

    // Create output channel
    outputChannel = vscode.window.createOutputChannel('Code Analyzer');

    // Create status bar item
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = '$(code) Code Analyzer';
    statusBarItem.tooltip = 'Python Code Analyzer';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Initialize diagnostics provider
    diagnosticsProvider = new DiagnosticsProvider();
    context.subscriptions.push(diagnosticsProvider);

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('codeAnalyzer.analyzeFile', analyzeFileCommand),
        vscode.commands.registerCommand('codeAnalyzer.analyzeDirectory', analyzeDirectoryCommand),
        vscode.commands.registerCommand('codeAnalyzer.securityScan', securityScanCommand),
        vscode.commands.registerCommand('codeAnalyzer.quickMetrics', quickMetricsCommand),
        vscode.commands.registerCommand('codeAnalyzer.ragQuery', ragQueryCommand)
    );

    // Auto-analyze on save if enabled
    const config = vscode.workspace.getConfiguration('codeAnalyzer');
    if (config.get<boolean>('autoAnalyzeOnSave')) {
        context.subscriptions.push(
            vscode.workspace.onDidSaveTextDocument(async (doc) => {
                if (doc.languageId === 'python') {
                    await analyzeFileQuiet(doc.uri.fsPath);
                }
            })
        );
    }
}

async function analyzeFileCommand(uri?: vscode.Uri) {
    const filePath = uri?.fsPath || vscode.window.activeTextEditor?.document.uri.fsPath;

    if (!filePath) {
        vscode.window.showWarningMessage('No file selected');
        return;
    }

    await runWithProgress('Analyzing file...', async () => {
        const result = await analyzer.analyzeFile(filePath);
        handleAnalysisResult(result, 'File Analysis');
    });
}

async function analyzeDirectoryCommand(uri?: vscode.Uri) {
    let dirPath = uri?.fsPath;

    if (!dirPath) {
        const folders = vscode.workspace.workspaceFolders;
        if (folders && folders.length > 0) {
            dirPath = folders[0].uri.fsPath;
        } else {
            vscode.window.showWarningMessage('No directory selected');
            return;
        }
    }

    await runWithProgress('Analyzing directory...', async () => {
        const result = await analyzer.analyzeDirectory(dirPath!);
        handleAnalysisResult(result, 'Directory Analysis');
    });
}

async function securityScanCommand(uri?: vscode.Uri) {
    const targetPath = uri?.fsPath ||
        vscode.window.activeTextEditor?.document.uri.fsPath ||
        vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;

    if (!targetPath) {
        vscode.window.showWarningMessage('No target selected');
        return;
    }

    await runWithProgress('Running security scan...', async () => {
        const result = await analyzer.securityScan(targetPath);
        handleAnalysisResult(result, 'Security Scan');
    });
}

async function quickMetricsCommand() {
    const filePath = vscode.window.activeTextEditor?.document.uri.fsPath;

    if (!filePath) {
        vscode.window.showWarningMessage('No file open');
        return;
    }

    await runWithProgress('Calculating metrics...', async () => {
        const result = await analyzer.quickMetrics(filePath);

        if (result.success && result.data) {
            const data = result.data;
            vscode.window.showInformationMessage(
                `📊 Metrics: Complexity=${data.cyclomatic_complexity || 'N/A'}, ` +
                `LOC=${data.lines_of_code || 'N/A'}, ` +
                `Maintainability=${data.maintainability_index?.toFixed(1) || 'N/A'}`
            );
        } else {
            vscode.window.showErrorMessage(result.error || 'Failed to get metrics');
        }
    });
}

async function ragQueryCommand() {
    const question = await vscode.window.showInputBox({
        prompt: 'Ask a question about your code',
        placeHolder: 'e.g., How does authentication work?'
    });

    if (!question) {
        return;
    }

    const projectPath = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!projectPath) {
        vscode.window.showWarningMessage('No workspace folder open');
        return;
    }

    await runWithProgress('Querying AI...', async () => {
        const result = await analyzer.ragQuery(question, projectPath);

        if (result.success && result.data) {
            outputChannel.clear();
            outputChannel.appendLine('🤖 AI Response:');
            outputChannel.appendLine('='.repeat(50));
            outputChannel.appendLine(result.data.answer || result.data.raw || JSON.stringify(result.data, null, 2));
            outputChannel.show();
        } else {
            vscode.window.showErrorMessage(result.error || 'AI query failed');
        }
    });
}

async function analyzeFileQuiet(filePath: string) {
    const result = await analyzer.analyzeFile(filePath);
    if (result.success && result.data) {
        diagnosticsProvider.updateDiagnostics(filePath, result.data);
    }
}

function handleAnalysisResult(result: analyzer.AnalysisResult, title: string) {
    if (result.success && result.data) {
        outputChannel.clear();
        outputChannel.appendLine(`📊 ${title}`);
        outputChannel.appendLine('='.repeat(50));

        const data = result.data;

        // Show summary
        if (data.total_files !== undefined) {
            outputChannel.appendLine(`Files analyzed: ${data.total_files}`);
        }
        if (data.total_issues !== undefined) {
            outputChannel.appendLine(`Total issues: ${data.total_issues}`);
        }

        // Show vulnerabilities
        if (data.vulnerabilities && data.vulnerabilities.length > 0) {
            outputChannel.appendLine('\n🔒 Security Issues:');
            for (const v of data.vulnerabilities) {
                outputChannel.appendLine(`  [${v.severity}] ${v.message}`);
                outputChannel.appendLine(`    File: ${path.basename(v.file_path)}:${v.line_number}`);
            }
        }

        // Show code smells
        if (data.code_smells && data.code_smells.length > 0) {
            outputChannel.appendLine('\n🔍 Code Smells:');
            for (const s of data.code_smells.slice(0, 10)) {
                outputChannel.appendLine(`  [${s.severity}] ${s.message}`);
            }
            if (data.code_smells.length > 10) {
                outputChannel.appendLine(`  ... and ${data.code_smells.length - 10} more`);
            }
        }

        // Show patterns
        if (data.design_patterns && data.design_patterns.length > 0) {
            outputChannel.appendLine('\n🎯 Design Patterns Found:');
            for (const p of data.design_patterns) {
                outputChannel.appendLine(`  ${p.pattern_type}: ${p.class_name}`);
            }
        }

        outputChannel.appendLine('\n' + '='.repeat(50));
        outputChannel.show();

        // Update diagnostics
        if (data.file_path) {
            diagnosticsProvider.updateDiagnostics(data.file_path, data);
        }

        vscode.window.showInformationMessage(`${title} complete. See Output panel.`);
    } else {
        vscode.window.showErrorMessage(result.error || 'Analysis failed');
    }
}

async function runWithProgress<T>(message: string, task: () => Promise<T>): Promise<T> {
    statusBarItem.text = '$(sync~spin) Analyzing...';

    try {
        return await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: message,
                cancellable: false
            },
            task
        );
    } finally {
        statusBarItem.text = '$(code) Code Analyzer';
    }
}

export function deactivate() {
    if (outputChannel) {
        outputChannel.dispose();
    }
}
