"""
Duplicate code detector.

Finds similar or duplicate code blocks using AST comparison.
"""

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import defaultdict

from analyzer.models.code_entities import Module, Function, Class
from analyzer.logging_config import get_logger

logger = get_logger("patterns.duplicates")


@dataclass
class CodeBlock:
    """A block of code for comparison."""
    file_path: str
    start_line: int
    end_line: int
    code: str
    ast_hash: str
    entity_name: Optional[str] = None


@dataclass
class DuplicateGroup:
    """A group of duplicate code blocks."""
    blocks: list[CodeBlock] = field(default_factory=list)
    similarity: float = 1.0
    line_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "count": len(self.blocks),
            "similarity": self.similarity,
            "lines": self.line_count,
            "locations": [
                {
                    "file": b.file_path,
                    "start_line": b.start_line,
                    "end_line": b.end_line,
                    "entity": b.entity_name,
                }
                for b in self.blocks
            ],
        }


class DuplicateDetector:
    """Detects duplicate code using AST comparison."""
    
    def __init__(
        self, 
        min_lines: int = 5,
        similarity_threshold: float = 0.8
    ):
        self.min_lines = min_lines
        self.similarity_threshold = similarity_threshold
    
    def detect(self, modules: list[Module]) -> list[DuplicateGroup]:
        """
        Detect duplicate code across modules.
        
        Args:
            modules: List of parsed modules
            
        Returns:
            List of duplicate groups
        """
        # Collect all code blocks
        blocks = self._collect_blocks(modules)
        
        # Group by hash
        duplicates = self._find_duplicates(blocks)
        
        return duplicates
    
    def detect_from_code(
        self, 
        codes: list[tuple[str, str]]  # [(code, filename), ...]
    ) -> list[DuplicateGroup]:
        """Detect duplicates from code strings."""
        blocks = []
        
        for code, filename in codes:
            blocks.extend(self._extract_blocks_from_code(code, filename))
        
        return self._find_duplicates(blocks)
    
    def _collect_blocks(self, modules: list[Module]) -> list[CodeBlock]:
        """Collect code blocks from modules."""
        blocks = []
        
        for module in modules:
            # Functions
            for func in module.functions:
                block = self._create_block_from_function(func, module.file_path)
                if block:
                    blocks.append(block)
            
            # Class methods
            for cls in module.classes:
                for method in cls.methods:
                    block = self._create_block_from_function(
                        method, module.file_path, f"{cls.name}.{method.name}"
                    )
                    if block:
                        blocks.append(block)
        
        return blocks
    
    def _create_block_from_function(
        self, 
        func: Function, 
        file_path: str,
        entity_name: Optional[str] = None
    ) -> Optional[CodeBlock]:
        """Create a code block from a function."""
        line_count = func.location.line_count
        
        if line_count < self.min_lines:
            return None
        
        # Create a normalized AST hash
        ast_hash = self._hash_function_ast(func)
        
        return CodeBlock(
            file_path=file_path,
            start_line=func.location.start_line,
            end_line=func.location.end_line,
            code="",  # We don't store the code for memory efficiency
            ast_hash=ast_hash,
            entity_name=entity_name or func.name,
        )
    
    def _hash_function_ast(self, func: Function) -> str:
        """Create a hash from function structure (ignoring names)."""
        # Create a normalized representation
        normalized = [
            len(func.parameters),
            len(func.calls),
            func.is_async,
            func.is_generator,
            func.location.line_count,
        ]
        
        # Include parameter types
        for param in func.parameters:
            normalized.append(param.type_annotation or "")
        
        # Include return type
        normalized.append(func.return_type or "")
        
        # Include calls (sorted for consistency)
        normalized.extend(sorted(func.calls))
        
        # Hash the normalized representation
        content = str(normalized).encode()
        return hashlib.md5(content).hexdigest()
    
    def _extract_blocks_from_code(
        self, 
        code: str, 
        filename: str
    ) -> list[CodeBlock]:
        """Extract code blocks from source code."""
        blocks = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return blocks
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                line_count = node.end_lineno - node.lineno + 1
                
                if line_count < self.min_lines:
                    continue
                
                ast_hash = self._hash_ast_node(node)
                
                blocks.append(CodeBlock(
                    file_path=filename,
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    code=ast.unparse(node) if hasattr(ast, 'unparse') else "",
                    ast_hash=ast_hash,
                    entity_name=node.name,
                ))
        
        return blocks
    
    def _hash_ast_node(self, node: ast.AST) -> str:
        """Hash an AST node, normalizing names."""
        def normalize(n: ast.AST) -> Any:
            """Normalize AST node for comparison."""
            if isinstance(n, ast.Name):
                # Normalize variable names
                return ("Name", "var")
            elif isinstance(n, ast.Constant):
                return ("Const", type(n.value).__name__)
            elif isinstance(n, ast.FunctionDef):
                return ("FuncDef", [normalize(c) for c in n.body])
            elif isinstance(n, ast.AsyncFunctionDef):
                return ("AsyncFuncDef", [normalize(c) for c in n.body])
            else:
                children = []
                for child in ast.iter_child_nodes(n):
                    children.append(normalize(child))
                return (type(n).__name__, children)
        
        normalized = normalize(node)
        return hashlib.md5(str(normalized).encode()).hexdigest()
    
    def _find_duplicates(self, blocks: list[CodeBlock]) -> list[DuplicateGroup]:
        """Find groups of duplicate blocks."""
        # Group blocks by hash
        hash_groups: dict[str, list[CodeBlock]] = defaultdict(list)
        
        for block in blocks:
            hash_groups[block.ast_hash].append(block)
        
        # Create duplicate groups
        duplicates = []
        
        for hash_value, group_blocks in hash_groups.items():
            if len(group_blocks) < 2:
                continue
            
            # Filter out same-file, overlapping blocks
            filtered = self._filter_overlapping(group_blocks)
            
            if len(filtered) < 2:
                continue
            
            avg_lines = sum(b.end_line - b.start_line + 1 for b in filtered) / len(filtered)
            
            duplicates.append(DuplicateGroup(
                blocks=filtered,
                similarity=1.0,  # Exact hash match
                line_count=int(avg_lines),
            ))
        
        # Sort by impact (line count * occurrence count)
        duplicates.sort(key=lambda d: d.line_count * len(d.blocks), reverse=True)
        
        return duplicates
    
    def _filter_overlapping(self, blocks: list[CodeBlock]) -> list[CodeBlock]:
        """Filter out overlapping blocks from the same file."""
        filtered = []
        
        for block in blocks:
            # Check if this overlaps with any already filtered block
            overlaps = False
            for existing in filtered:
                if block.file_path == existing.file_path:
                    # Check for line overlap
                    if (block.start_line <= existing.end_line and 
                        block.end_line >= existing.start_line):
                        overlaps = True
                        break
            
            if not overlaps:
                filtered.append(block)
        
        return filtered


def detect_duplicates(
    modules: list[Module],
    min_lines: int = 5,
    similarity_threshold: float = 0.8
) -> list[DuplicateGroup]:
    """
    Detect duplicate code in modules.
    
    Args:
        modules: List of parsed modules
        min_lines: Minimum lines for a code block
        similarity_threshold: Minimum similarity (0.0-1.0)
        
    Returns:
        List of duplicate groups
    """
    detector = DuplicateDetector(min_lines, similarity_threshold)
    return detector.detect(modules)
