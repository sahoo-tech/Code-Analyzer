

import ast
import keyword
from typing import Set
from collections import Counter

from analyzer.models.metrics import HalsteadMetrics
from analyzer.logging_config import get_logger

logger = get_logger("metrics.halstead")


class HalsteadCalculator:

    # Python operators
    OPERATORS = {
        # Arithmetic
        '+', '-', '*', '/', '//', '%', '**', '@',
        # Comparison
        '==', '!=', '<', '>', '<=', '>=',
        # Logical
        'and', 'or', 'not',
        # Bitwise
        '&', '|', '^', '~', '<<', '>>',
        # Assignment
        '=', '+=', '-=', '*=', '/=', '//=', '%=', '**=',
        '&=', '|=', '^=', '<<=', '>>=', '@=',
        # Membership & Identity
        'in', 'is',
        # Other
        '.', ',', ':', ';', '->', ':=',
        '(', ')', '[', ']', '{', '}',
    }
    
    # Keywords that act as operators
    OPERATOR_KEYWORDS = {
        'if', 'else', 'elif', 'for', 'while', 'with',
        'try', 'except', 'finally', 'raise', 'assert',
        'def', 'class', 'return', 'yield', 'lambda',
        'import', 'from', 'as', 'pass', 'break', 'continue',
        'global', 'nonlocal', 'del', 'async', 'await',
        'and', 'or', 'not', 'in', 'is',
    }
    
    def calculate(self, code: str) -> HalsteadMetrics:
   
        try:
            tree = ast.parse(code)
            return self._analyze_tree(tree)
        except SyntaxError:
            return HalsteadMetrics()
    
    def _analyze_tree(self, tree: ast.AST) -> HalsteadMetrics:
        """Analyze AST tree for operators and operands."""
        operators: Counter = Counter()
        operands: Counter = Counter()
        
        for node in ast.walk(tree):
            self._process_node(node, operators, operands)
        
        return HalsteadMetrics(
            n1=len(operators),  # Distinct operators
            n2=len(operands),   # Distinct operands
            N1=sum(operators.values()),  # Total operators
            N2=sum(operands.values()),   # Total operands
        )
    
    def _process_node(
        self, 
        node: ast.AST, 
        operators: Counter,
        operands: Counter
    ) -> None:
        """Process a single AST node."""
        # Binary operations
        if isinstance(node, ast.BinOp):
            op_name = type(node.op).__name__
            operators[op_name] += 1
        
        # Unary operations
        elif isinstance(node, ast.UnaryOp):
            op_name = type(node.op).__name__
            operators[op_name] += 1
        
        # Boolean operations
        elif isinstance(node, ast.BoolOp):
            op_name = type(node.op).__name__
            operators[op_name] += len(node.values) - 1
        
        # Comparison operations
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                operators[type(op).__name__] += 1
        
        # Augmented assignment
        elif isinstance(node, ast.AugAssign):
            op_name = type(node.op).__name__
            operators[f"Aug{op_name}"] += 1
        
        # Regular assignment
        elif isinstance(node, ast.Assign):
            operators["Assign"] += 1
        
        # Annotated assignment
        elif isinstance(node, ast.AnnAssign):
            operators["AnnAssign"] += 1
        
        # Function call
        elif isinstance(node, ast.Call):
            operators["Call"] += 1
        
        # Attribute access
        elif isinstance(node, ast.Attribute):
            operators["."] += 1
            operands[node.attr] += 1
        
        # Subscript
        elif isinstance(node, ast.Subscript):
            operators["Subscript"] += 1
        
        # Keywords/control flow
        elif isinstance(node, ast.If):
            operators["if"] += 1
        elif isinstance(node, ast.While):
            operators["while"] += 1
        elif isinstance(node, ast.For):
            operators["for"] += 1
        elif isinstance(node, ast.AsyncFor):
            operators["async_for"] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            operators["def"] += 1
            operands[node.name] += 1
        elif isinstance(node, ast.ClassDef):
            operators["class"] += 1
            operands[node.name] += 1
        elif isinstance(node, ast.Return):
            operators["return"] += 1
        elif isinstance(node, ast.Yield):
            operators["yield"] += 1
        elif isinstance(node, ast.YieldFrom):
            operators["yield_from"] += 1
        elif isinstance(node, ast.Try):
            operators["try"] += 1
        elif isinstance(node, ast.ExceptHandler):
            operators["except"] += 1
        elif isinstance(node, ast.With):
            operators["with"] += 1
        elif isinstance(node, ast.AsyncWith):
            operators["async_with"] += 1
        elif isinstance(node, ast.Raise):
            operators["raise"] += 1
        elif isinstance(node, ast.Assert):
            operators["assert"] += 1
        elif isinstance(node, ast.Import):
            operators["import"] += 1
        elif isinstance(node, ast.ImportFrom):
            operators["from_import"] += 1
        elif isinstance(node, ast.Lambda):
            operators["lambda"] += 1
        elif isinstance(node, ast.Pass):
            operators["pass"] += 1
        elif isinstance(node, ast.Break):
            operators["break"] += 1
        elif isinstance(node, ast.Continue):
            operators["continue"] += 1
        
        # Operands
        elif isinstance(node, ast.Name):
            if node.id not in keyword.kwlist:
                operands[node.id] += 1
        
        elif isinstance(node, ast.Constant):
            operands[repr(node.value)] += 1
        
        # Comprehensions
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            operators["comprehension"] += 1
        elif isinstance(node, ast.DictComp):
            operators["dict_comprehension"] += 1


def calculate_halstead(code: str) -> HalsteadMetrics:
 
    return HalsteadCalculator().calculate(code)
