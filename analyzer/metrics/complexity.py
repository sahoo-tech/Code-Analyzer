

import ast
from typing import Union, Optional

from analyzer.models.code_entities import Function, Method, Class, Module
from analyzer.models.metrics import ComplexityMetrics
from analyzer.logging_config import get_logger

logger = get_logger("metrics.complexity")


class CyclomaticComplexityCalculator:

    
    # Decision point nodes that add to complexity
    DECISION_NODES = (
        ast.If,
        ast.While,
        ast.For,
        ast.AsyncFor,
        ast.ExceptHandler,
        ast.With,
        ast.AsyncWith,
        ast.Assert,
    )
    
    def calculate(self, code: str) -> int:
        """Calculate cyclomatic complexity for code string."""
        try:
            tree = ast.parse(code)
            return self._visit(tree)
        except SyntaxError:
            return 1
    
    def calculate_for_node(self, node: ast.AST) -> int:
        """Calculate complexity for an AST node."""
        return self._visit(node)
    
    def _visit(self, node: ast.AST) -> int:
        """Visit AST node and count decision points."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Decision points
            if isinstance(child, self.DECISION_NODES):
                complexity += 1
            
            # Boolean operators (and, or)
            elif isinstance(child, ast.BoolOp):
                # Each and/or adds a decision point
                complexity += len(child.values) - 1
            
            # Comprehensions
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # Count comprehension generators
                comprehension_complexity = 0
                for generator in child.generators:
                    comprehension_complexity += 1  # for the loop
                    comprehension_complexity += len(generator.ifs)  # for conditions
                complexity += comprehension_complexity
            
            # Ternary expression (a if b else c)
            elif isinstance(child, ast.IfExp):
                complexity += 1
            
            # Match statement (Python 3.10+)
            elif isinstance(child, ast.Match):
                complexity += len(child.cases) - 1
        
        return complexity


class CognitiveComplexityCalculator:

    
    def calculate(self, code: str) -> int:
        """Calculate cognitive complexity for code string."""
        try:
            tree = ast.parse(code)
            return self._calculate_cognitive(tree)
        except SyntaxError:
            return 0
    
    def calculate_for_node(self, node: ast.AST) -> int:
        """Calculate complexity for an AST node."""
        return self._calculate_cognitive(node)
    
    def _calculate_cognitive(self, node: ast.AST, nesting: int = 0) -> int:
        """Calculate cognitive complexity recursively."""
        complexity = 0
        
        for child in ast.iter_child_nodes(node):
            complexity += self._process_node(child, nesting)
        
        return complexity
    
    def _process_node(self, node: ast.AST, nesting: int) -> int:
        """Process a single node for cognitive complexity."""
        complexity = 0
        new_nesting = nesting
        
        # Structural complexity: increment and nest
        if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            complexity += 1 + nesting  # Base + nesting penalty
            new_nesting = nesting + 1
            
            # Handle elif chains (only adds 1, not nesting penalty)
            if isinstance(node, ast.If):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.If) and child in node.orelse:
                        # This is an elif, already counted differently
                        pass
        
        # Exception handling
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1 + nesting
            new_nesting = nesting + 1
        
        # Try block itself (only its except handlers add complexity)
        elif isinstance(node, ast.Try):
            new_nesting = nesting + 1
        
        # Nesting increments (no base increment)
        elif isinstance(node, (ast.Lambda, ast.With, ast.AsyncWith)):
            new_nesting = nesting + 1
        
        # Recursion (function calling itself) - add 1
        # This would need context tracking
        
        # Breaks in linear flow
        elif isinstance(node, (ast.Break, ast.Continue)):
            complexity += 1
        
        # Boolean sequences - add for each changed operator
        elif isinstance(node, ast.BoolOp):
            # Count operator changes
            complexity += 1  # First operator
        
        # Ternary
        elif isinstance(node, ast.IfExp):
            complexity += 1 + nesting
        
        # Nested functions/classes without lambda
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if nesting > 0:
                new_nesting = nesting + 1
        
        # Recurse into children
        for child in ast.iter_child_nodes(node):
            complexity += self._process_node(child, new_nesting)
        
        return complexity


class NestingDepthCalculator:
    """Calculates maximum nesting depth."""
    
    def calculate(self, code: str) -> int:
        """Calculate max nesting depth for code string."""
        try:
            tree = ast.parse(code)
            return self._max_depth(tree, 0)
        except SyntaxError:
            return 0
    
    def _max_depth(self, node: ast.AST, current: int) -> int:
        """Calculate maximum depth recursively."""
        child_depth = current
        
        # Nodes that increase nesting
        if isinstance(node, (
            ast.If, ast.While, ast.For, ast.AsyncFor,
            ast.With, ast.AsyncWith, ast.Try,
            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        )):
            child_depth = current + 1
        
        max_depth = child_depth
        
        for child in ast.iter_child_nodes(node):
            depth = self._max_depth(child, child_depth)
            max_depth = max(max_depth, depth)
        
        return max_depth


def calculate_complexity(
    code: Optional[str] = None,
    node: Optional[ast.AST] = None
) -> ComplexityMetrics:
 
    if code is None and node is None:
        return ComplexityMetrics()
    
    cyclomatic_calc = CyclomaticComplexityCalculator()
    cognitive_calc = CognitiveComplexityCalculator()
    nesting_calc = NestingDepthCalculator()
    
    if code:
        return ComplexityMetrics(
            cyclomatic=cyclomatic_calc.calculate(code),
            cognitive=cognitive_calc.calculate(code),
            max_nesting_depth=nesting_calc.calculate(code),
        )
    else:
        return ComplexityMetrics(
            cyclomatic=cyclomatic_calc.calculate_for_node(node),
            cognitive=cognitive_calc.calculate_for_node(node),
            max_nesting_depth=0,  # Would need code for this
        )
