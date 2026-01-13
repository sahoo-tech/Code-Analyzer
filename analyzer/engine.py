"""
Main analyzer engine.

Orchestrates all analysis modules to provide comprehensive code analysis.
"""

import json
from pathlib import Path
from typing import Any, Optional, Union
from dataclasses import dataclass, field

from analyzer.parsers import FileParser
from analyzer.metrics import calculate_complexity, calculate_loc, calculate_halstead, calculate_maintainability
from analyzer.dependencies import analyze_imports, build_call_graph, build_module_graph
from analyzer.patterns import detect_design_patterns, detect_anti_patterns, detect_code_smells, detect_dead_code, detect_duplicates
from analyzer.security import scan_vulnerabilities, detect_secrets
from analyzer.ai import format_for_ai, summarize_project, QueryInterface
from analyzer.models.code_entities import Module
from analyzer.models.metrics import ProjectMetrics, FileMetrics, QualityScore
from analyzer.config import get_config, AnalyzerConfig
from analyzer.logging_config import get_logger, configure_logging
from analyzer.utils import read_file, validate_path

logger = get_logger("engine")


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    # Parsed modules
    modules: list[Module] = field(default_factory=list)
    
    # Metrics
    project_metrics: Optional[ProjectMetrics] = None
    
    # Dependencies
    call_graph: Optional[Any] = None
    module_graph: Optional[Any] = None
    circular_dependencies: list[list[str]] = field(default_factory=list)
    
    # Patterns
    design_patterns: list[Any] = field(default_factory=list)
    anti_patterns: list[Any] = field(default_factory=list)
    code_smells: list[Any] = field(default_factory=list)
    dead_code: list[Any] = field(default_factory=list)
    duplicates: list[Any] = field(default_factory=list)
    
    # Security
    vulnerabilities: list[Any] = field(default_factory=list)
    secrets: list[Any] = field(default_factory=list)
    
    # Meta
    file_count: int = 0
    total_lines: int = 0
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "overview": {
                "file_count": self.file_count,
                "total_lines": self.total_lines,
                "module_count": len(self.modules),
            },
            "modules": [m.to_dict() for m in self.modules],
            "metrics": self.project_metrics.to_dict() if self.project_metrics else {},
            "dependencies": {
                "call_graph": self.call_graph.to_dict() if self.call_graph else {},
                "module_graph": self.module_graph.to_dict() if self.module_graph else {},
                "circular_dependencies": self.circular_dependencies,
            },
            "patterns": {
                "design_patterns": [p.to_dict() for p in self.design_patterns],
                "anti_patterns": [p.to_dict() for p in self.anti_patterns],
                "code_smells": [s.to_dict() for s in self.code_smells],
                "dead_code": [d.to_dict() for d in self.dead_code],
                "duplicates": [d.to_dict() for d in self.duplicates],
            },
            "security": {
                "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
                "secrets": [s.to_dict() for s in self.secrets],
            },
            "errors": self.errors,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def get_summary(self) -> dict:
        """Get summary of analysis results."""
        return {
            "files_analyzed": self.file_count,
            "total_lines": self.total_lines,
            "classes": sum(len(m.classes) for m in self.modules),
            "functions": sum(len(m.functions) for m in self.modules),
            "design_patterns_found": len(self.design_patterns),
            "anti_patterns_found": len(self.anti_patterns),
            "code_smells_found": len(self.code_smells),
            "security_issues": len(self.vulnerabilities) + len(self.secrets),
            "circular_dependencies": len(self.circular_dependencies),
        }


class CodeAnalyzer:
    """
    Main code analyzer engine.
    
    Orchestrates all analysis modules to provide comprehensive
    code analysis for AI systems.
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """
        Initialize the analyzer.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or get_config()
        self.parser = FileParser(self.config)
        
        # Configure logging
        configure_logging(
            level=self.config.logging.level,
            log_file=self.config.logging.file,
        )
    
    def analyze_file(self, path: Union[str, Path]) -> AnalysisResult:
        """
        Analyze a single file.
        
        Args:
            path: Path to the file
            
        Returns:
            AnalysisResult with analysis data
        """
        path = validate_path(path)
        
        logger.info(f"Analyzing file: {path}")
        
        result = AnalysisResult()
        
        try:
            module = self.parser.parse_file(path)
            result.modules = [module]
            result.file_count = 1
            result.total_lines = module.location.line_count if module.location else 0
            
            self._analyze_modules(result)
            
        except Exception as e:
            logger.error(f"Error analyzing file: {e}")
            result.errors.append(str(e))
        
        return result
    
    def analyze_directory(
        self, 
        path: Union[str, Path],
        recursive: bool = True,
    ) -> AnalysisResult:
        """
        Analyze a directory of files.
        
        Args:
            path: Path to the directory
            recursive: Whether to analyze subdirectories
            
        Returns:
            AnalysisResult with analysis data
        """
        path = validate_path(path)
        
        logger.info(f"Analyzing directory: {path}")
        
        result = AnalysisResult()
        
        try:
            # Parse all files
            modules = self.parser.parse_directory(
                path,
                recursive=recursive,
            )
            
            result.modules = modules
            result.file_count = len(modules)
            result.total_lines = sum(
                m.location.line_count if m.location else 0 
                for m in modules
            )
            
            self._analyze_modules(result, project_root=path)
            
        except Exception as e:
            logger.error(f"Error analyzing directory: {e}")
            result.errors.append(str(e))
        
        return result
    
    def analyze_code(
        self, 
        code: str, 
        filename: str = "<string>"
    ) -> AnalysisResult:
        """
        Analyze code from a string.
        
        Args:
            code: Source code string
            filename: Optional filename for context
            
        Returns:
            AnalysisResult with analysis data
        """
        logger.info(f"Analyzing code: {filename}")
        
        result = AnalysisResult()
        
        try:
            module = self.parser.parse_code(code, filename=filename)
            result.modules = [module]
            result.file_count = 1
            result.total_lines = len(code.splitlines())
            
            self._analyze_modules(result)
            
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            result.errors.append(str(e))
        
        return result
    
    def _analyze_modules(
        self, 
        result: AnalysisResult,
        project_root: Optional[Path] = None
    ) -> None:
        """Run all analysis on parsed modules."""
        modules = result.modules
        
        if not modules:
            return
        
        # Import analysis
        logger.debug("Analyzing imports...")
        for module in modules:
            analyze_imports(module, project_root)
        
        # Dependency graphs
        if self.config.dependencies.detect_circular:
            logger.debug("Building dependency graphs...")
            result.call_graph = build_call_graph(modules)
            result.module_graph = build_module_graph(modules, project_root)
            result.circular_dependencies = result.module_graph.find_circular_dependencies()
        
        # Pattern detection
        logger.debug("Detecting patterns...")
        if self.config.patterns.detect_design_patterns:
            result.design_patterns = detect_design_patterns(modules)
        
        if self.config.patterns.detect_anti_patterns:
            result.anti_patterns = detect_anti_patterns(modules)
        
        if self.config.patterns.detect_code_smells:
            result.code_smells = detect_code_smells(modules)
        
        if self.config.patterns.detect_dead_code:
            result.dead_code = detect_dead_code(modules)
        
        if self.config.patterns.detect_duplicates:
            result.duplicates = detect_duplicates(modules)
        
        # Security analysis
        logger.debug("Running security analysis...")
        if self.config.security.check_sql_injection or self.config.security.check_dangerous_functions:
            result.vulnerabilities = scan_vulnerabilities(modules)
        
        if self.config.security.check_hardcoded_secrets:
            result.secrets = detect_secrets(modules)
        
        logger.info(f"Analysis complete: {len(modules)} files processed")
    
    def get_ai_summary(
        self, 
        result: AnalysisResult,
        format_type: str = "json"
    ) -> str:
        """
        Get AI-optimized summary of analysis.
        
        Args:
            result: Analysis result
            format_type: Output format ("json" or "markdown")
            
        Returns:
            Formatted summary string
        """
        return format_for_ai(result.to_dict(), format_type)
    
    def query(self, result: AnalysisResult, query_string: str) -> dict:
        """
        Query analysis results using natural language.
        
        Args:
            result: Analysis result
            query_string: Natural language query
            
        Returns:
            Query result as dictionary
        """
        interface = QueryInterface(result.modules)
        query_result = interface.query(query_string)
        return query_result.to_dict()
