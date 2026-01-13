"""
Dead code detector.

Identifies unused code:
- Unused imports
- Unused variables
- Unused functions
- Unreachable code
"""

import ast
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from analyzer.models.code_entities import Module
from analyzer.logging_config import get_logger

logger = get_logger("patterns.dead_code")


class DeadCodeType(Enum):
    """Types of dead code."""
    UNUSED_IMPORT = "unused_import"
    UNUSED_VARIABLE = "unused_variable"
    UNUSED_FUNCTION = "unused_function"
    UNUSED_CLASS = "unused_class"
    UNREACHABLE_CODE = "unreachable_code"
    UNUSED_PARAMETER = "unused_parameter"


@dataclass
class DeadCode:
    """Detected dead code."""
    code_type: DeadCodeType
    name: str
    file_path: str
    line_number: int
    confidence: float  # 0.0 to 1.0
    
    def to_dict(self) -> dict:
        return {
            "type": self.code_type.value,
            "name": self.name,
            "file": self.file_path,
            "line": self.line_number,
            "confidence": self.confidence,
        }


class DeadCodeDetector:
    """Detects dead/unused code."""
    
    def detect(self, modules: list[Module]) -> list[DeadCode]:
        """
        Detect dead code in modules.
        
        Args:
            modules: List of parsed modules
            
        Returns:
            List of dead code findings
        """
        dead_code = []
        
        for module in modules:
            dead_code.extend(self._analyze_module(module))
        
        return dead_code
    
    def detect_from_code(self, code: str, file_path: str = "<string>") -> list[DeadCode]:
        """Detect dead code from code string."""
        try:
            tree = ast.parse(code)
            return self._analyze_ast(tree, file_path)
        except SyntaxError:
            return []
    
    def _analyze_module(self, module: Module) -> list[DeadCode]:
        """Analyze a module for dead code."""
        dead_code = []
        
        # Collect all used names in the module
        used_names = self._collect_used_names(module)
        
        # Check unused imports
        for imp in module.imports:
            used_name = imp.used_name
            if used_name not in used_names:
                # Skip common patterns like typing imports
                if imp.module in ('typing', '__future__'):
                    continue
                
                dead_code.append(DeadCode(
                    code_type=DeadCodeType.UNUSED_IMPORT,
                    name=used_name,
                    file_path=module.file_path,
                    line_number=imp.location.start_line if imp.location else 1,
                    confidence=0.9,
                ))
        
        # Check unused functions (only in module scope, high false positive rate)
        for func in module.functions:
            if func.name.startswith('_') and func.name not in used_names:
                dead_code.append(DeadCode(
                    code_type=DeadCodeType.UNUSED_FUNCTION,
                    name=func.name,
                    file_path=module.file_path,
                    line_number=func.location.start_line,
                    confidence=0.6,  # Lower confidence
                ))
        
        return dead_code
    
    def _collect_used_names(self, module: Module) -> set[str]:
        """Collect all names used in a module."""
        used = set()
        
        # Add all function calls
        for func in module.functions:
            used.update(func.calls)
        
        for cls in module.classes:
            # Add base classes
            used.update(cls.bases)
            
            # Add method calls
            for method in cls.methods:
                used.update(method.calls)
            
            # Add class variable types
            for var in cls.class_variables:
                if var.type_annotation:
                    used.update(self._extract_type_names(var.type_annotation))
        
        # Add function type annotations
        for func in module.functions:
            for param in func.parameters:
                if param.type_annotation:
                    used.update(self._extract_type_names(param.type_annotation))
            if func.return_type:
                used.update(self._extract_type_names(func.return_type))
        
        return used
    
    def _extract_type_names(self, type_str: str) -> set[str]:
        """Extract type names from a type annotation string."""
        import re
        # Extract identifiers from type annotations
        return set(re.findall(r'\b([A-Z][a-zA-Z0-9_]*)\b', type_str))
    
    def _analyze_ast(self, tree: ast.AST, file_path: str) -> list[DeadCode]:
        """Analyze AST for dead code patterns."""
        dead_code = []
        
        for node in ast.walk(tree):
            # Unreachable code after return/raise/break/continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                dead = self._find_unreachable_code(node, file_path)
                dead_code.extend(dead)
        
        return dead_code
    
    def _find_unreachable_code(
        self, 
        func_node: ast.AST, 
        file_path: str
    ) -> list[DeadCode]:
        """Find unreachable code after control flow statements."""
        dead_code = []
        
        def check_body(body: list[ast.stmt]) -> None:
            for i, stmt in enumerate(body):
                # Check if this is a terminating statement
                if isinstance(stmt, (ast.Return, ast.Raise)):
                    # Check if there's code after this statement
                    if i < len(body) - 1:
                        next_stmt = body[i + 1]
                        dead_code.append(DeadCode(
                            code_type=DeadCodeType.UNREACHABLE_CODE,
                            name="<unreachable>",
                            file_path=file_path,
                            line_number=next_stmt.lineno,
                            confidence=0.95,
                        ))
                
                # Recurse into compound statements
                if isinstance(stmt, (ast.If, ast.For, ast.While, ast.With)):
                    if hasattr(stmt, 'body'):
                        check_body(stmt.body)
                    if hasattr(stmt, 'orelse'):
                        check_body(stmt.orelse)
                elif isinstance(stmt, ast.Try):
                    check_body(stmt.body)
                    for handler in stmt.handlers:
                        check_body(handler.body)
                    check_body(stmt.finalbody)
        
        check_body(func_node.body)
        return dead_code


def detect_dead_code(modules: list[Module]) -> list[DeadCode]:
    """
    Detect dead code in modules.
    
    Args:
        modules: List of parsed modules
        
    Returns:
        List of dead code findings
    """
    detector = DeadCodeDetector()
    return detector.detect(modules)
