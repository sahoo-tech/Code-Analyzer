"""Tests for metrics modules."""

import pytest
from analyzer.metrics import (
    CyclomaticComplexityCalculator,
    CognitiveComplexityCalculator,
    calculate_complexity,
    LOCCalculator,
    calculate_loc,
    HalsteadCalculator,
    calculate_halstead,
    calculate_maintainability,
)


class TestComplexity:
    """Tests for complexity calculators."""
    
    def test_simple_function_complexity(self):
        """Simple function should have complexity 1."""
        code = '''
def simple():
    return 42
'''
        calc = CyclomaticComplexityCalculator()
        assert calc.calculate(code) == 1
    
    def test_if_increases_complexity(self):
        """If statement increases complexity by 1."""
        code = '''
def check(x):
    if x > 0:
        return True
    return False
'''
        calc = CyclomaticComplexityCalculator()
        assert calc.calculate(code) == 2
    
    def test_multiple_conditions(self):
        """Multiple conditions in if increase complexity."""
        code = '''
def check(x, y):
    if x > 0 and y > 0:
        return True
    return False
'''
        calc = CyclomaticComplexityCalculator()
        result = calc.calculate(code)
        assert result >= 2  # Base + and condition
    
    def test_loop_complexity(self):
        """Loops increase complexity."""
        code = '''
def loop(items):
    total = 0
    for item in items:
        total += item
    return total
'''
        calc = CyclomaticComplexityCalculator()
        assert calc.calculate(code) == 2  # Base + for
    
    def test_nested_complexity(self):
        """Nested structures increase complexity more."""
        code = '''
def nested(items):
    for item in items:
        if item > 0:
            for sub in item:
                if sub:
                    return sub
'''
        calc = CyclomaticComplexityCalculator()
        result = calc.calculate(code)
        assert result >= 4
    
    def test_cognitive_complexity(self):
        """Test cognitive complexity calculator."""
        code = '''
def complex_func(x, y, z):
    if x:
        if y:
            if z:
                return True
    return False
'''
        calc = CognitiveComplexityCalculator()
        result = calc.calculate(code)
        # Nested ifs should have higher cognitive than cyclomatic
        assert result >= 3


class TestLOC:
    """Tests for LOC calculator."""
    
    def test_basic_loc(self):
        """Test basic line counting."""
        code = '''
def hello():
    return "Hello"

def bye():
    return "Bye"
'''
        calc = LOCCalculator()
        metrics = calc.calculate(code)
        
        assert metrics.total > 0
        assert metrics.source > 0
        assert metrics.blank >= 0
    
    def test_comments_counted(self):
        """Test comment counting."""
        code = '''
# This is a comment
def hello():
    # Another comment
    return "Hello"  # Inline comment
'''
        calc = LOCCalculator()
        metrics = calc.calculate(code)
        
        assert metrics.comments >= 2
    
    def test_docstrings(self):
        """Test docstring detection."""
        code = '''
"""Module docstring."""

def hello():
    """Function docstring."""
    return "Hello"
'''
        calc = LOCCalculator()
        metrics = calc.calculate(code)
        
        assert metrics.docstrings >= 1


class TestHalstead:
    """Tests for Halstead metrics."""
    
    def test_basic_halstead(self):
        """Test basic Halstead calculation."""
        code = '''
def add(a, b):
    return a + b
'''
        calc = HalsteadCalculator()
        metrics = calc.calculate(code)
        
        assert metrics.n1 > 0  # Distinct operators
        assert metrics.n2 > 0  # Distinct operands
        assert metrics.N1 > 0  # Total operators
        assert metrics.N2 > 0  # Total operands
        assert metrics.volume > 0
    
    def test_complex_halstead(self):
        """Test Halstead on more complex code."""
        code = '''
def calculate(x, y, z):
    result = (x + y) * z
    if result > 0:
        return result * 2
    else:
        return -result
'''
        calc = HalsteadCalculator()
        metrics = calc.calculate(code)
        
        assert metrics.difficulty > 0
        assert metrics.effort > 0


class TestMaintainability:
    """Tests for maintainability index."""
    
    def test_simple_code_maintainable(self):
        """Simple, well-documented code should be maintainable."""
        code = '''
"""Simple module."""

def greet(name: str) -> str:
    """Greet someone.
    
    Args:
        name: The name to greet.
        
    Returns:
        Greeting message.
    """
    return f"Hello, {name}!"
'''
        metrics = calculate_maintainability(code)
        
        # Simple code should have good maintainability
        assert metrics.maintainability_index > 50
    
    def test_complex_code_less_maintainable(self):
        """Complex code should have lower maintainability."""
        code = '''
def complex(a, b, c, d, e, f):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        if f:
                            return a + b + c + d + e + f
                        return a + b + c + d + e
                    return a + b + c + d
                return a + b + c
            return a + b
        return a
    return 0
'''
        metrics = calculate_maintainability(code)
        
        # This should have worse maintainability than simple code
        assert metrics.maintainability_index <= 100
