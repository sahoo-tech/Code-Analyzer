"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from textwrap import dedent


@pytest.fixture
def sample_code():
    """Simple Python code for testing."""
    return dedent('''
        """Sample module for testing."""
        
        import os
        from typing import Optional
        
        CONSTANT = 42
        
        
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}!"
        
        
        def calculate(a: int, b: int) -> int:
            """Calculate sum of two numbers."""
            return a + b
        
        
        class Person:
            """Represents a person."""
            
            def __init__(self, name: str, age: int):
                self.name = name
                self.age = age
            
            def introduce(self) -> str:
                """Introduce the person."""
                return f"I am {self.name}, {self.age} years old."
    ''').strip()


@pytest.fixture
def complex_code():
    """Complex Python code with various patterns."""
    return dedent('''
        """Complex module with various patterns."""
        
        import os
        import sys
        from abc import ABC, abstractmethod
        from typing import Optional, List
        
        PASSWORD = "secret123"  # Hardcoded secret
        
        
        class Singleton:
            """Singleton pattern implementation."""
            _instance = None
            
            def __new__(cls):
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                return cls._instance
            
            @classmethod
            def get_instance(cls):
                return cls()
        
        
        class AbstractHandler(ABC):
            """Abstract base class."""
            
            @abstractmethod
            def handle(self, data):
                pass
        
        
        class GodClass:
            """A class that does too much."""
            
            def method1(self): pass
            def method2(self): pass
            def method3(self): pass
            def method4(self): pass
            def method5(self): pass
            def method6(self): pass
            def method7(self): pass
            def method8(self): pass
            def method9(self): pass
            def method10(self): pass
            def method11(self): pass
            def method12(self): pass
            def method13(self): pass
            def method14(self): pass
            def method15(self): pass
            def method16(self): pass
            def method17(self): pass
            def method18(self): pass
            def method19(self): pass
            def method20(self): pass
            def method21(self): pass
        
        
        def vulnerable_function(user_input):
            """Contains security vulnerabilities."""
            os.system(user_input)  # Command injection
            eval(user_input)  # Eval usage
            query = "SELECT * FROM users WHERE id = %s" % user_input  # SQL injection
            return query
        
        
        async def async_function():
            """Async function example."""
            return await some_coroutine()
    ''').strip()


@pytest.fixture
def temp_python_file(tmp_path, sample_code):
    """Create a temporary Python file."""
    file_path = tmp_path / "test_module.py"
    file_path.write_text(sample_code, encoding="utf-8")
    return file_path


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure."""
    # Main module
    (tmp_path / "main.py").write_text(dedent('''
        """Main module."""
        from utils import helper
        
        def main():
            return helper()
    ''').strip())
    
    # Utils module
    (tmp_path / "utils.py").write_text(dedent('''
        """Utility functions."""
        
        def helper():
            return "helped"
    ''').strip())
    
    # Package
    pkg_dir = tmp_path / "mypackage"
    pkg_dir.mkdir()
    
    (pkg_dir / "__init__.py").write_text('"""Package init."""')
    
    (pkg_dir / "module.py").write_text(dedent('''
        """Package module."""
        
        class MyClass:
            def method(self):
                pass
    ''').strip())
    
    return tmp_path
