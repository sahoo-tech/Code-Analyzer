
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

    analyzer = CodeAnalyzer(config)
    return analyzer.analyze_file(path)


def analyze_directory(
    path: Union[str, Path],
    recursive: bool = True,
    config: Optional[AnalyzerConfig] = None
) -> AnalysisResult:

    analyzer = CodeAnalyzer(config)
    return analyzer.analyze_directory(path, recursive=recursive)


def analyze_code(
    code: str,
    filename: str = "<string>",
    config: Optional[AnalyzerConfig] = None
) -> AnalysisResult:

    analyzer = CodeAnalyzer(config)
    return analyzer.analyze_code(code, filename=filename)


def parse_file(path: Union[str, Path]) -> Module:

    parser = FileParser()
    return parser.parse_file(path)


def parse_code(code: str, filename: str = "<string>") -> Module:
  
    parser = FileParser()
    return parser.parse_code(code, filename=filename)


def get_metrics(path: Union[str, Path]) -> dict[str, Any]:

    result = analyze_file(path) if Path(path).is_file() else analyze_directory(path)
    return result.project_metrics.to_dict() if result.project_metrics else {}


def check_security(path: Union[str, Path]) -> list[dict]:

    result = analyze_file(path) if Path(path).is_file() else analyze_directory(path)
    
    findings = []
    for vuln in result.vulnerabilities:
        findings.append(vuln.to_dict())
    for secret in result.secrets:
        findings.append(secret.to_dict())
    
    return findings


def find_patterns(path: Union[str, Path]) -> dict[str, list[dict]]:

    result = analyze_file(path) if Path(path).is_file() else analyze_directory(path)
    
    return {
        "design_patterns": [p.to_dict() for p in result.design_patterns],
        "anti_patterns": [p.to_dict() for p in result.anti_patterns],
        "code_smells": [s.to_dict() for s in result.code_smells],
    }


def query_code(path: Union[str, Path], query: str) -> dict:
 
    analyzer = CodeAnalyzer()
    analysis = analyzer.analyze_file(path) if Path(path).is_file() else analyzer.analyze_directory(path)
    return analyzer.query(analysis, query)


def get_summary(path: Union[str, Path], format_type: str = "markdown") -> str:

    analyzer = CodeAnalyzer()
    result = analyzer.analyze_file(path) if Path(path).is_file() else analyzer.analyze_directory(path)
    return analyzer.get_ai_summary(result, format_type)
