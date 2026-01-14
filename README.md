# Enterprise Code Analyzer

An enterprise-level Python code analyzer designed to help AI systems understand and analyze code with unprecedented depth and clarity.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

### Deep Code Analysis
- **AST Parsing**: Full abstract syntax tree analysis with semantic context
- **Entity Extraction**: Classes, functions, methods, variables, imports
- **Type Analysis**: Type annotation parsing and inference

### Comprehensive Metrics
- **Cyclomatic Complexity**: McCabe complexity measurement
- **Cognitive Complexity**: SonarSource-style cognitive complexity
- **Halstead Metrics**: Volume, difficulty, effort, estimated bugs
- **Maintainability Index**: Microsoft-style MI calculation
- **Lines of Code**: Total, source, comments, blank, docstrings

### Dependency Intelligence
- **Import Analysis**: Categorizes stdlib, third-party, and local imports
- **Call Graphs**: Function/method call relationship mapping
- **Module Dependencies**: Inter-module dependency graphs
- **Circular Detection**: Automatic circular dependency detection

### Pattern Detection
- **Design Patterns**: Singleton, Factory, Observer, Decorator, Strategy, Builder
- **Anti-Patterns**: God Class, Long Method, Long Parameter List
- **Code Smells**: Magic numbers, empty except, mutable defaults
- **Dead Code**: Unused imports, unreachable code
- **Duplicates**: AST-based duplicate detection

### Security Analysis
- **Vulnerability Scanning**: SQL injection, command injection, eval/exec
- **Secret Detection**: Hardcoded passwords, API keys, tokens
- **Insecure Patterns**: Pickle, unsafe YAML, shell=True

### AI Integration
- **LLM-Optimized Output**: JSON and Markdown formatters
- **Code Summarization**: Concise summaries for context windows
- **Natural Language Queries**: Query code using plain English

### RAG (Retrieval-Augmented Generation)
- **Semantic Code Search**: Vector-based similarity search over code entities
- **Natural Language Q&A**: Ask questions about your codebase in plain English
- **Multi-Provider Support**: OpenAI, Anthropic, Google, or local embeddings
- **Hybrid Retrieval**: Combines semantic and keyword search with reranking
- **Persistent Index**: ChromaDB-powered index survives restarts


## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/code-analyzer.git
cd code-analyzer

# Install in development mode
pip install -e ".[dev]"

# Or install dependencies directly
pip install -r requirements.txt
```

## 🔧 Quick Start

### Command Line

```bash
# Analyze a file
code-analyzer analyze myfile.py

# Analyze a directory
code-analyzer analyze ./src --output results.json

# Get a summary
code-analyzer summary ./src

# Query the codebase
code-analyzer query ./src "find all async functions"

# Initialize config
code-analyzer init
```

### Python API

```python
from analyzer import analyze_file, analyze_directory, analyze_code

# Analyze a file
result = analyze_file("mymodule.py")
print(result.get_summary())

# Analyze a directory
result = analyze_directory("./src")
print(f"Found {len(result.vulnerabilities)} security issues")

# Analyze code string
code = '''
def hello(name: str) -> str:
    return f"Hello, {name}!"
'''
result = analyze_code(code)

# Get JSON output
print(result.to_json())

# Query the code
from analyzer import CodeAnalyzer
analyzer = CodeAnalyzer()
result = analyzer.analyze_directory("./src")
query_result = analyzer.query(result, "find classes inheriting from ABC")
```

### Using Individual Components

```python
from analyzer.parsers import PythonParser
from analyzer.metrics import calculate_complexity, calculate_maintainability
from analyzer.security import scan_vulnerabilities

# Parse code
parser = PythonParser()
module = parser.parse_code(code)

# Get metrics
complexity = calculate_complexity(code)
print(f"Cyclomatic: {complexity.cyclomatic}")

# Check security
vulnerabilities = scan_vulnerabilities([module])
for vuln in vulnerabilities:
    print(f"{vuln.severity}: {vuln.message}")
```

## ⚙️ Configuration

Create a `.code-analyzer.yaml` file:

```yaml
parser:
  max_file_size_mb: 10
  encoding: utf-8

metrics:
  complexity_threshold_high: 20
  max_function_lines: 50

patterns:
  detect_design_patterns: true
  detect_anti_patterns: true
  detect_code_smells: true

security:
  check_sql_injection: true
  check_hardcoded_secrets: true

ai:
  max_context_tokens: 8000
  output_format: json

logging:
  level: INFO
```

Or use environment variables:

```bash
export CODE_ANALYZER_LOG_LEVEL=DEBUG
export CODE_ANALYZER_MAX_WORKERS=8
```

## 📊 Output Examples

### JSON Output
```json
{
  "overview": {
    "file_count": 10,
    "total_lines": 1500,
    "classes": 15,
    "functions": 45
  },
  "quality": {
    "maintainability_index": 72.5,
    "rating": "B"
  },
  "security": {
    "vulnerabilities": [...],
    "secrets": [...]
  }
}
```

### Markdown Summary
```markdown
# Code Analysis Report

## Overview
- **Files Analyzed:** 10
- **Total Lines:** 1500
- **Classes:** 15
- **Functions:** 45

## Quality Metrics
- **Overall Rating:** B
- **Maintainability Index:** 72.5
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=analyzer --cov-report=term-missing

# Run specific tests
pytest tests/test_parsers.py -v
```

## 📁 Project Structure

```
code-analyzer/
├── analyzer/
│   ├── __init__.py          # Package initialization
│   ├── engine.py             # Main orchestration engine
│   ├── cli.py                # Command-line interface
│   ├── api.py                # Public API
│   ├── config.py             # Configuration management
│   ├── exceptions.py         # Custom exceptions
│   ├── utils.py              # Utilities
│   ├── parsers/              # Code parsing
│   ├── models/               # Data models
│   ├── metrics/              # Code metrics
│   ├── dependencies/         # Dependency analysis
│   ├── patterns/             # Pattern detection
│   ├── security/             # Security analysis
│   └── ai/                   # AI integration
├── tests/                    # Test suite
├── docs/                     # Documentation
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🤖 AI Integration

This analyzer is specifically designed to provide AI systems with structured, actionable insights:

1. **Token-Aware Summaries**: Generates summaries that fit within LLM context windows
2. **Structured Output**: JSON format optimized for programmatic consumption
3. **Natural Language Queries**: Query code using plain English
4. **Context-Rich Metadata**: Includes location, metrics, and relationships

Example AI workflow:
```python
from analyzer import CodeAnalyzer

analyzer = CodeAnalyzer()
result = analyzer.analyze_directory("./project")

# Get AI-optimized summary
summary = analyzer.get_ai_summary(result, "markdown")

# Send to LLM with structured context
llm_prompt = f"""
Analyze this codebase:

{summary}

What improvements would you suggest?
"""
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.



1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request
