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
exports.ResultsPanel = void 0;
const vscode = __importStar(require("vscode"));
/**
 * WebView panel for displaying detailed analysis results
 */
class ResultsPanel {
    static currentPanel;
    panel;
    disposables = [];
    constructor(panel) {
        this.panel = panel;
        this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    }
    static createOrShow(extensionUri, title, data) {
        const column = vscode.window.activeTextEditor
            ? vscode.window.activeTextEditor.viewColumn
            : undefined;
        if (ResultsPanel.currentPanel) {
            ResultsPanel.currentPanel.panel.reveal(column);
            ResultsPanel.currentPanel.update(title, data);
            return ResultsPanel.currentPanel;
        }
        const panel = vscode.window.createWebviewPanel('codeAnalyzerResults', 'Code Analyzer Results', column || vscode.ViewColumn.One, {
            enableScripts: true,
            retainContextWhenHidden: true
        });
        ResultsPanel.currentPanel = new ResultsPanel(panel);
        ResultsPanel.currentPanel.update(title, data);
        return ResultsPanel.currentPanel;
    }
    update(title, data) {
        this.panel.title = title;
        this.panel.webview.html = this.getHtmlContent(title, data);
    }
    getHtmlContent(title, data) {
        const vulnerabilities = data.vulnerabilities || [];
        const codeSmells = data.code_smells || [];
        const designPatterns = data.design_patterns || [];
        const antiPatterns = data.anti_patterns || [];
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 20px;
            line-height: 1.6;
        }
        h1 { color: var(--vscode-textLink-foreground); border-bottom: 1px solid var(--vscode-panel-border); padding-bottom: 10px; }
        h2 { color: var(--vscode-textPreformat-foreground); margin-top: 24px; }
        .section { margin-bottom: 24px; }
        .item {
            background: var(--vscode-editor-inactiveSelectionBackground);
            padding: 12px;
            margin: 8px 0;
            border-radius: 6px;
            border-left: 4px solid var(--vscode-textLink-foreground);
        }
        .item.critical { border-left-color: #f44336; }
        .item.high { border-left-color: #ff9800; }
        .item.medium { border-left-color: #ffeb3b; }
        .item.low { border-left-color: #4caf50; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
        }
        .badge.critical { background: #f44336; color: white; }
        .badge.high { background: #ff9800; color: black; }
        .badge.medium { background: #ffeb3b; color: black; }
        .badge.low { background: #4caf50; color: white; }
        .badge.pattern { background: #2196f3; color: white; }
        .file-link { color: var(--vscode-textLink-foreground); font-size: 12px; }
        .recommendation { 
            margin-top: 8px; 
            padding: 8px; 
            background: var(--vscode-textBlockQuote-background);
            border-radius: 4px;
            font-size: 13px;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }
        .summary-card {
            background: var(--vscode-button-secondaryBackground);
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        .summary-card .number { font-size: 28px; font-weight: bold; color: var(--vscode-textLink-foreground); }
        .summary-card .label { font-size: 12px; color: var(--vscode-descriptionForeground); }
        .empty { color: var(--vscode-descriptionForeground); font-style: italic; }
    </style>
</head>
<body>
    <h1>📊 ${title}</h1>
    
    <div class="summary">
        <div class="summary-card">
            <div class="number">${vulnerabilities.length}</div>
            <div class="label">Security Issues</div>
        </div>
        <div class="summary-card">
            <div class="number">${codeSmells.length}</div>
            <div class="label">Code Smells</div>
        </div>
        <div class="summary-card">
            <div class="number">${designPatterns.length}</div>
            <div class="label">Design Patterns</div>
        </div>
        <div class="summary-card">
            <div class="number">${antiPatterns.length}</div>
            <div class="label">Anti-Patterns</div>
        </div>
    </div>
    
    ${this.renderVulnerabilities(vulnerabilities)}
    ${this.renderCodeSmells(codeSmells)}
    ${this.renderPatterns(designPatterns, antiPatterns)}
</body>
</html>`;
    }
    renderVulnerabilities(vulnerabilities) {
        if (vulnerabilities.length === 0) {
            return `<div class="section"><h2>🔒 Security Issues</h2><p class="empty">✅ No security issues found!</p></div>`;
        }
        const items = vulnerabilities.map(v => `
            <div class="item ${v.severity?.toLowerCase()}">
                <span class="badge ${v.severity?.toLowerCase()}">${v.severity}</span>
                <strong>${v.type || 'Issue'}</strong>: ${v.message}
                <div class="file-link">📄 ${v.file_path}:${v.line_number}</div>
                ${v.recommendation ? `<div class="recommendation">💡 ${v.recommendation}</div>` : ''}
            </div>
        `).join('');
        return `<div class="section"><h2>🔒 Security Issues</h2>${items}</div>`;
    }
    renderCodeSmells(smells) {
        if (smells.length === 0) {
            return `<div class="section"><h2>🔍 Code Smells</h2><p class="empty">✅ No code smells found!</p></div>`;
        }
        const items = smells.slice(0, 20).map(s => `
            <div class="item ${s.severity?.toLowerCase()}">
                <span class="badge ${s.severity?.toLowerCase()}">${s.severity}</span>
                ${s.message}
                <div class="file-link">📄 ${s.file_path}:${s.line_number}</div>
            </div>
        `).join('');
        const more = smells.length > 20 ? `<p>... and ${smells.length - 20} more</p>` : '';
        return `<div class="section"><h2>🔍 Code Smells</h2>${items}${more}</div>`;
    }
    renderPatterns(design, anti) {
        let html = '<div class="section"><h2>🎯 Patterns</h2>';
        if (design.length > 0) {
            html += '<h3>✨ Design Patterns</h3>';
            html += design.map(p => `
                <div class="item">
                    <span class="badge pattern">${p.pattern_type}</span>
                    <strong>${p.class_name}</strong>
                    <div class="file-link">Confidence: ${(p.confidence * 100).toFixed(0)}%</div>
                </div>
            `).join('');
        }
        if (anti.length > 0) {
            html += '<h3>⚠️ Anti-Patterns</h3>';
            html += anti.map(p => `
                <div class="item medium">
                    <span class="badge medium">${p.pattern_type}</span>
                    ${p.description}
                    ${p.suggestion ? `<div class="recommendation">💡 ${p.suggestion}</div>` : ''}
                </div>
            `).join('');
        }
        if (design.length === 0 && anti.length === 0) {
            html += '<p class="empty">No patterns detected.</p>';
        }
        html += '</div>';
        return html;
    }
    dispose() {
        ResultsPanel.currentPanel = undefined;
        this.panel.dispose();
        while (this.disposables.length) {
            const disposable = this.disposables.pop();
            if (disposable) {
                disposable.dispose();
            }
        }
    }
}
exports.ResultsPanel = ResultsPanel;
//# sourceMappingURL=resultsPanel.js.map