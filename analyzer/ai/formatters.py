

import json
from dataclasses import dataclass
from typing import Any, Optional
from abc import ABC, abstractmethod

from analyzer.models.code_entities import Module, Class, Function
from analyzer.models.metrics import QualityScore
from analyzer.logging_config import get_logger

logger = get_logger("ai.formatters")


@dataclass
class FormatterConfig:
    """Configuration for AI formatters."""
    max_tokens: int = 8000
    include_source: bool = True
    include_metrics: bool = True
    include_dependencies: bool = True
    include_issues: bool = True
    detail_level: str = "detailed"  # minimal, summary, detailed


class AIFormatter(ABC):
    """Base class for AI-friendly formatters."""
    
    def __init__(self, config: Optional[FormatterConfig] = None):
        self.config = config or FormatterConfig()
    
    @abstractmethod
    def format(self, data: Any) -> str:
        """Format data for AI consumption."""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Average of 4 chars per token for English
        return len(text) // 4


class JSONFormatter(AIFormatter):
    """Formats analysis results as structured JSON."""
    
    def format(self, data: Any) -> str:
        """Format data as JSON."""
        if hasattr(data, 'to_dict'):
            data = data.to_dict()
        
        return json.dumps(data, indent=2, default=str)
    
    def format_module(self, module: Module) -> str:
        """Format a module for AI analysis."""
        result = {
            "module": {
                "name": module.name,
                "file": module.file_path,
                "summary": self._create_summary(module),
            },
        }
        
        if self.config.include_source:
            result["structure"] = self._format_structure(module)
        
        if self.config.include_metrics:
            result["metrics"] = self._format_metrics(module)
        
        if self.config.include_dependencies:
            result["dependencies"] = self._format_dependencies(module)
        
        return json.dumps(result, indent=2)
    
    def format_analysis(self, analysis: dict) -> str:
        """Format complete analysis results."""
        result = {
            "overview": analysis.get("overview", {}),
            "modules": [],
            "quality": analysis.get("quality", {}),
            "issues": [],
        }
        
        # Add module summaries
        for module_data in analysis.get("modules", []):
            result["modules"].append({
                "name": module_data.get("name"),
                "path": module_data.get("path"),
                "classes": len(module_data.get("classes", [])),
                "functions": len(module_data.get("functions", [])),
            })
        
        # Add issues summary
        for issue in analysis.get("issues", [])[:20]:  # Limit to top 20
            result["issues"].append({
                "type": issue.get("type"),
                "severity": issue.get("severity"),
                "message": issue.get("message"),
                "location": issue.get("location"),
            })
        
        return json.dumps(result, indent=2)
    
    def _create_summary(self, module: Module) -> dict:
        """Create module summary."""
        return {
            "classes": len(module.classes),
            "functions": len(module.functions),
            "imports": len(module.imports),
            "lines": module.location.line_count if module.location else 0,
            "has_docstring": module.docstring is not None,
        }
    
    def _format_structure(self, module: Module) -> dict:
        """Format module structure."""
        return {
            "classes": [
                {
                    "name": cls.name,
                    "bases": cls.bases,
                    "methods": [m.name for m in cls.methods],
                    "is_abstract": cls.is_abstract,
                }
                for cls in module.classes
            ],
            "functions": [
                {
                    "name": func.name,
                    "params": [p.name for p in func.parameters],
                    "returns": func.return_type,
                    "is_async": func.is_async,
                }
                for func in module.functions
            ],
        }
    
    def _format_metrics(self, module: Module) -> dict:
        """Format module metrics."""
        return {
            "complexity": "N/A",  # Would be calculated separately
            "maintainability": "N/A",
        }
    
    def _format_dependencies(self, module: Module) -> dict:
        """Format module dependencies."""
        return {
            "imports": [
                {
                    "module": imp.module,
                    "name": imp.name,
                    "is_relative": imp.is_relative,
                }
                for imp in module.imports
            ],
        }


class MarkdownFormatter(AIFormatter):
    """Formats analysis results as Markdown."""
    
    def format(self, data: Any) -> str:
        """Format data as Markdown."""
        if isinstance(data, Module):
            return self.format_module(data)
        elif isinstance(data, dict):
            return self._dict_to_markdown(data)
        return str(data)
    
    def format_module(self, module: Module) -> str:
        """Format a module as Markdown."""
        lines = [
            f"# Module: {module.name}",
            "",
            f"**File:** `{module.file_path}`",
            "",
        ]
        
        # Docstring
        if module.docstring:
            lines.extend([
                "## Description",
                "",
                module.docstring.summary,
                "",
            ])
        
        # Summary
        lines.extend([
            "## Summary",
            "",
            f"- **Classes:** {len(module.classes)}",
            f"- **Functions:** {len(module.functions)}",
            f"- **Imports:** {len(module.imports)}",
            "",
        ])
        
        # Classes
        if module.classes:
            lines.extend(["## Classes", ""])
            for cls in module.classes:
                lines.extend(self._format_class_md(cls))
        
        # Functions
        if module.functions:
            lines.extend(["## Functions", ""])
            for func in module.functions:
                lines.extend(self._format_function_md(func))
        
        return "\n".join(lines)
    
    def format_analysis(self, analysis: dict) -> str:
        """Format analysis results as Markdown report."""
        lines = [
            "# Code Analysis Report",
            "",
        ]
        
        # Overview
        overview = analysis.get("overview", {})
        if overview:
            lines.extend([
                "## Overview",
                "",
                f"- **Files Analyzed:** {overview.get('file_count', 'N/A')}",
                f"- **Total Lines:** {overview.get('total_lines', 'N/A')}",
                f"- **Classes:** {overview.get('total_classes', 'N/A')}",
                f"- **Functions:** {overview.get('total_functions', 'N/A')}",
                "",
            ])
        
        # Quality
        quality = analysis.get("quality", {})
        if quality:
            lines.extend([
                "## Quality Metrics",
                "",
                f"- **Overall Rating:** {quality.get('rating', 'N/A')}",
                f"- **Maintainability Index:** {quality.get('maintainability', 'N/A')}",
                "",
            ])
        
        # Issues
        issues = analysis.get("issues", [])
        if issues:
            lines.extend([
                "## Issues Found",
                "",
            ])
            
            for issue in issues[:10]:  # Top 10 issues
                severity = issue.get("severity", "info").upper()
                message = issue.get("message", "")
                location = issue.get("location", "")
                lines.append(f"- **[{severity}]** {message} ({location})")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_class_md(self, cls: Class) -> list[str]:
        """Format class as Markdown."""
        lines = [
            f"### `{cls.name}`",
            "",
        ]
        
        if cls.bases:
            lines.append(f"**Inherits:** {', '.join(cls.bases)}")
            lines.append("")
        
        if cls.docstring:
            lines.append(cls.docstring.summary)
            lines.append("")
        
        if cls.methods:
            lines.append("**Methods:**")
            for method in cls.methods[:10]:  # Limit to 10
                lines.append(f"- `{method.name}()`")
            lines.append("")
        
        return lines
    
    def _format_function_md(self, func: Function) -> list[str]:
        """Format function as Markdown."""
        lines = [
            f"### `{func.name}`",
            "",
        ]
        
        # Signature
        params = ", ".join(p.name for p in func.parameters)
        ret = f" -> {func.return_type}" if func.return_type else ""
        lines.append(f"```python\ndef {func.name}({params}){ret}\n```")
        lines.append("")
        
        if func.docstring:
            lines.append(func.docstring.summary)
            lines.append("")
        
        return lines
    
    def _dict_to_markdown(self, data: dict, level: int = 1) -> str:
        """Convert dict to Markdown."""
        lines = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{'#' * level} {key}")
                lines.append("")
                lines.append(self._dict_to_markdown(value, level + 1))
            elif isinstance(value, list):
                lines.append(f"**{key}:**")
                for item in value[:10]:
                    lines.append(f"- {item}")
                lines.append("")
            else:
                lines.append(f"- **{key}:** {value}")
        
        return "\n".join(lines)


def format_for_ai(
    data: Any,
    format_type: str = "json",
    config: Optional[FormatterConfig] = None
) -> str:

    if format_type == "json":
        formatter = JSONFormatter(config)
    elif format_type == "markdown":
        formatter = MarkdownFormatter(config)
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    return formatter.format(data)
