"""
Code smell detector.

Identifies code smells indicating potential problems:
- Complex conditionals
- Magic numbers
- Commented code
- Empty except blocks
- Mutable default arguments
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from analyzer.models.code_entities import Module, Class, Function
from analyzer.logging_config import get_logger

logger = get_logger("patterns.code_smells")


class SmellType(Enum):
    """Types of code smells."""
    MAGIC_NUMBER = "magic_number"
    COMPLEX_CONDITIONAL = "complex_conditional"
    COMMENTED_CODE = "commented_code"
    EMPTY_EXCEPT = "empty_except"
    MUTABLE_DEFAULT = "mutable_default"
    MISSING_DOCSTRING = "missing_docstring"
    TOO_MANY_RETURNS = "too_many_returns"
    INCONSISTENT_NAMING = "inconsistent_naming"
    BARE_EXCEPT = "bare_except"
    UNUSED_VARIABLE = "unused_variable"
    GLOBAL_VARIABLE = "global_variable"


@dataclass
class CodeSmell:
    """A detected code smell."""
    smell_type: SmellType
    message: str
    file_path: str
    line_number: int
    severity: str  # "info", "warning", "error"
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "type": self.smell_type.value,
            "message": self.message,
            "file": self.file_path,
            "line": self.line_number,
            "severity": self.severity,
            "snippet": self.code_snippet,
            "suggestion": self.suggestion,
        }


class CodeSmellDetector:
    """Detects code smells in Python code."""
    
    # Magic number exclusions
    ALLOWED_NUMBERS = {0, 1, -1, 2, 10, 100, 1000}
    
    def detect(self, modules: list[Module]) -> list[CodeSmell]:
        """
        Detect code smells in modules.
        
        Args:
            modules: List of parsed modules
            
        Returns:
            List of detected code smells
        """
        smells = []
        
        for module in modules:
            smells.extend(self._analyze_module(module))
        
        return smells
    
    def detect_from_code(self, code: str, file_path: str = "<string>") -> list[CodeSmell]:
        """Detect code smells directly from code string."""
        smells = []
        
        try:
            tree = ast.parse(code)
            smells.extend(self._analyze_ast(tree, code, file_path))
        except SyntaxError:
            pass
        
        return smells
    
    def _analyze_module(self, module: Module) -> list[CodeSmell]:
        """Analyze a module for code smells."""
        smells = []
        
        # Check for missing module docstring
        if not module.docstring:
            smells.append(CodeSmell(
                smell_type=SmellType.MISSING_DOCSTRING,
                message="Module missing docstring",
                file_path=module.file_path,
                line_number=1,
                severity="info",
                suggestion="Add a module-level docstring explaining the module's purpose",
            ))
        
        # Check classes
        for cls in module.classes:
            smells.extend(self._analyze_class(cls, module.file_path))
        
        # Check functions
        for func in module.functions:
            smells.extend(self._analyze_function(func, module.file_path))
        
        return smells
    
    def _analyze_class(self, cls: Class, file_path: str) -> list[CodeSmell]:
        """Analyze a class for code smells."""
        smells = []
        
        # Missing class docstring
        if not cls.docstring and not cls.is_dataclass:
            smells.append(CodeSmell(
                smell_type=SmellType.MISSING_DOCSTRING,
                message=f"Class '{cls.name}' missing docstring",
                file_path=file_path,
                line_number=cls.location.start_line,
                severity="info",
                suggestion="Add a class docstring explaining its purpose",
            ))
        
        # Check methods
        for method in cls.methods:
            smells.extend(self._analyze_function(method, file_path))
        
        return smells
    
    def _analyze_function(self, func: Function, file_path: str) -> list[CodeSmell]:
        """Analyze a function for code smells."""
        smells = []
        
        # Missing docstring for public functions
        from analyzer.models.code_entities import Visibility
        if not func.docstring and func.visibility == Visibility.PUBLIC:
            if func.name not in ('__init__', '__str__', '__repr__'):
                smells.append(CodeSmell(
                    smell_type=SmellType.MISSING_DOCSTRING,
                    message=f"Public function '{func.name}' missing docstring",
                    file_path=file_path,
                    line_number=func.location.start_line,
                    severity="info",
                    suggestion="Add a docstring describing parameters, returns, and behavior",
                ))
        
        # Mutable default arguments
        for param in func.parameters:
            if param.default_value:
                default = param.default_value
                if default.startswith(('[', '{')) or default in ('list()', 'dict()', 'set()'):
                    smells.append(CodeSmell(
                        smell_type=SmellType.MUTABLE_DEFAULT,
                        message=f"Mutable default argument: {param.name}={default}",
                        file_path=file_path,
                        line_number=func.location.start_line,
                        severity="warning",
                        suggestion="Use None as default and initialize inside function",
                    ))
        
        return smells
    
    def _analyze_ast(self, tree: ast.AST, code: str, file_path: str) -> list[CodeSmell]:
        """Analyze AST for code smells."""
        smells = []
        lines = code.splitlines()
        
        for node in ast.walk(tree):
            # Magic numbers
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in self.ALLOWED_NUMBERS:
                    # Check if it's not in an assignment to a constant
                    smells.append(CodeSmell(
                        smell_type=SmellType.MAGIC_NUMBER,
                        message=f"Magic number: {node.value}",
                        file_path=file_path,
                        line_number=node.lineno,
                        severity="info",
                        suggestion="Extract to named constant",
                    ))
            
            # Empty except blocks
            elif isinstance(node, ast.ExceptHandler):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    smells.append(CodeSmell(
                        smell_type=SmellType.EMPTY_EXCEPT,
                        message="Empty except block",
                        file_path=file_path,
                        line_number=node.lineno,
                        severity="warning",
                        suggestion="Handle the exception or add a comment explaining why it's ignored",
                    ))
                
                # Bare except
                if node.type is None:
                    smells.append(CodeSmell(
                        smell_type=SmellType.BARE_EXCEPT,
                        message="Bare except clause catches all exceptions",
                        file_path=file_path,
                        line_number=node.lineno,
                        severity="warning",
                        suggestion="Catch specific exceptions instead",
                    ))
            
            # Global statements
            elif isinstance(node, ast.Global):
                for name in node.names:
                    smells.append(CodeSmell(
                        smell_type=SmellType.GLOBAL_VARIABLE,
                        message=f"Use of global variable: {name}",
                        file_path=file_path,
                        line_number=node.lineno,
                        severity="warning",
                        suggestion="Consider using class attributes or passing as parameters",
                    ))
            
            # Too many returns
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.Return))
                if return_count > 5:
                    smells.append(CodeSmell(
                        smell_type=SmellType.TOO_MANY_RETURNS,
                        message=f"Function has {return_count} return statements",
                        file_path=file_path,
                        line_number=node.lineno,
                        severity="info",
                        suggestion="Consider refactoring to reduce return points",
                    ))
        
        # Check for commented code
        smells.extend(self._detect_commented_code(code, file_path))
        
        return smells
    
    def _detect_commented_code(self, code: str, file_path: str) -> list[CodeSmell]:
        """Detect commented-out code."""
        smells = []
        
        # Patterns that suggest commented code
        code_patterns = [
            r'#\s*(def\s+\w+|class\s+\w+|import\s+|from\s+)',
            r'#\s*(if\s+|for\s+|while\s+|return\s+)',
            r'#\s*(\w+\s*=\s*|print\()',
        ]
        
        for i, line in enumerate(code.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                for pattern in code_patterns:
                    if re.match(pattern, stripped):
                        smells.append(CodeSmell(
                            smell_type=SmellType.COMMENTED_CODE,
                            message="Commented-out code detected",
                            file_path=file_path,
                            line_number=i,
                            severity="info",
                            code_snippet=stripped[:50],
                            suggestion="Remove dead code or restore if needed",
                        ))
                        break
        
        return smells


def detect_code_smells(modules: list[Module]) -> list[CodeSmell]:
    """
    Detect code smells in modules.
    
    Args:
        modules: List of parsed modules
        
    Returns:
        List of detected code smells
    """
    detector = CodeSmellDetector()
    return detector.detect(modules)
