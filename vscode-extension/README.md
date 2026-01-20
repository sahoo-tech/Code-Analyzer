# Python Code Analyzer - VS Code Extension

AI-Enhanced Static Analysis for Python directly in VS Code.

## Features

- **📂 Analyze File** - Analyze any Python file for issues
- **📁 Analyze Directory** - Scan entire projects
- **🔒 Security Scan** - Find vulnerabilities and hardcoded secrets
- **📊 Quick Metrics** - Get complexity and maintainability scores
- **🤖 Ask AI** - Query your code using natural language (requires API key)

## Installation

### From Source (Development)

1. Open the `vscode-extension` folder in VS Code
2. Press `F5` to launch Extension Development Host
3. Test the extension in the new VS Code window

### Package as VSIX

```bash
cd vscode-extension
npm run package
```

Then install the `.vsix` file via VS Code → Extensions → Install from VSIX.

## Requirements

- **Python 3.8+** with the analyzer package installed:
  ```bash
  pip install -e .
  ```
- **Node.js 18+** (for building)

## Usage

### Commands

Open Command Palette (`Ctrl+Shift+P`) and type:
- `Code Analyzer: Analyze File`
- `Code Analyzer: Analyze Directory`
- `Code Analyzer: Security Scan`
- `Code Analyzer: Quick Metrics`
- `Code Analyzer: Ask AI About Code`

### Context Menu

Right-click on:
- **Python files** → Analyze File, Quick Metrics
- **Folders** → Analyze Directory

### Inline Diagnostics

Vulnerabilities and code smells appear as inline warnings in the editor.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `codeAnalyzer.pythonPath` | `python` | Path to Python executable |
| `codeAnalyzer.showInlineWarnings` | `true` | Show inline warnings |
| `codeAnalyzer.autoAnalyzeOnSave` | `false` | Auto-analyze on save |

## Development

```bash
# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Watch for changes
npm run watch

# Package extension
npm run package
```

## License

MIT
