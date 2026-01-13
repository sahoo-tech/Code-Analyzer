"""
Public API for the Code Analyzer.

Provides simple functions for common analysis tasks.
"""

from pathlib import Path
from typing import Union, Optional, Any

from analyzer.engine import CodeAnalyzer, AnalysisResult
from analyzer.config import AnalyzerConfig
from analyzer.parsers import FileParser
from analyzer.models.code_entities import Module


def analyze_file(
    path: Union[str, Path],
    config: Optional[AnalyzerConfig] = None
) -> AnalysisResult:
    """
    Analyze a single Python file.
    
    Args:
        path: Path to the Python file
        config: Optional configuration
        
    Returns:
        AnalysisResult with complete analysis
        
    Example:
        >>> result = analyze_file("my_module.py")
        >>> print(result.get_summary())
    """
    analyzer = CodeAnalyzer(config)
    return analyzer.analyze_file(path)


def analyze_directory(
    path: Union[str, Path],
    recursive: bool = True,
    config: Optional[AnalyzerConfig] = None
) -> AnalysisResult:
    """
    Analyze a directory of Python files.
    
    Args:
        path: Path to the directory
        recursive: Whether to include subdirectories
        config: Optional configuration
        
    Returns:
        AnalysisResult with complete analysis
        
    Example:
        >>> result = analyze_directory("./src")
        >>> print(f"Found {len(result.vulnerabilities)} security issues")
    """
    analyzer = CodeAnalyzer(config)
    return analyzer.analyze_directory(path, recursive=recursive)


def analyze_code(
    code: str,
    filename: str = "<string>",
    config: Optional[AnalyzerConfig] = None
) -> AnalysisResult:
    """
    Analyze Python code from a string.
    
    Args:
        code: Python source code
        filename: Optional filename for context
        config: Optional configuration
        
    Returns:
        AnalysisResult with complete analysis
        
    Example:
        >>> code = '''
        ... def hello(name):
        ...     print(f"Hello, {name}!")
        ... '''
        >>> result = analyze_code(code)
    """
    analyzer = CodeAnalyzer(config)
    return analyzer.analyze_code(code, filename=filename)


def parse_file(path: Union[str, Path]) -> Module:
    """
    Parse a Python file into a Module object.
    
    Args:
        path: Path to the Python file
        
    Returns:
        Module object with parsed code structure
        
    Example:
        >>> module = parse_file("my_module.py")
        >>> print(f"Classes: {[c.name for c in module.classes]}")
    """
    parser = FileParser()
    return parser.parse_file(path)


def parse_code(code: str, filename: str = "<string>") -> Module:
    """
    Parse Python code from a string.
    
    Args:
        code: Python source code
        filename: Optional filename for context
        
    Returns:
        Module object with parsed code structure
    """
    parser = FileParser()
    return parser.parse_code(code, filename=filename)


def get_metrics(path: Union[str, Path]) -> dict[str, Any]:
    """
    Get code metrics for a file or directory.
    
    Args:
        path: Path to file or directory
        
    Returns:
        Dictionary with metrics
        
    Example:
        >>> metrics = get_metrics("./src")
        >>> print(f"Maintainability: {metrics['maintainability']}")
    """
    result = analyze_file(path) if Path(path).is_file() else analyze_directory(path)
    return result.project_metrics.to_dict() if result.project_metrics else {}


def check_security(path: Union[str, Path]) -> list[dict]:
    """
    Check for security issues in code.
    
    Args:
        path: Path to file or directory
        
    Returns:
        List of security findings
        
    Example:
        >>> issues = check_security("./src")
        >>> for issue in issues:
        ...     print(f"{issue['severity']}: {issue['message']}")
    """
    result = analyze_file(path) if Path(path).is_file() else analyze_directory(path)
    
    findings = []
    for vuln in result.vulnerabilities:
        findings.append(vuln.to_dict())
    for secret in result.secrets:
        findings.append(secret.to_dict())
    
    return findings


def find_patterns(path: Union[str, Path]) -> dict[str, list[dict]]:
    """
    Find patterns and anti-patterns in code.
    
    Args:
        path: Path to file or directory
        
    Returns:
        Dictionary with pattern findings
        
    Example:
        >>> patterns = find_patterns("./src")
        >>> print(f"Singletons: {len(patterns['design_patterns'])}")
    """
    result = analyze_file(path) if Path(path).is_file() else analyze_directory(path)
    
    return {
        "design_patterns": [p.to_dict() for p in result.design_patterns],
        "anti_patterns": [p.to_dict() for p in result.anti_patterns],
        "code_smells": [s.to_dict() for s in result.code_smells],
    }


def query_code(path: Union[str, Path], query: str) -> dict:
    """
    Query code using natural language.
    
    Args:
        path: Path to file or directory
        query: Natural language query
        
    Returns:
        Query result
        
    Example:
        >>> result = query_code("./src", "find all async functions")
        >>> print(f"Found {result['count']} matches")
    """
    analyzer = CodeAnalyzer()
    analysis = analyzer.analyze_file(path) if Path(path).is_file() else analyzer.analyze_directory(path)
    return analyzer.query(analysis, query)


def get_summary(path: Union[str, Path], format_type: str = "markdown") -> str:
    """
    Get AI-optimized summary of code.
    
    Args:
        path: Path to file or directory
        format_type: Output format ("json" or "markdown")
        
    Returns:
        Formatted summary string
        
    Example:
        >>> summary = get_summary("./src", "markdown")
        >>> print(summary)
    """
    analyzer = CodeAnalyzer()
    result = analyzer.analyze_file(path) if Path(path).is_file() else analyzer.analyze_directory(path)
    return analyzer.get_ai_summary(result, format_type)
