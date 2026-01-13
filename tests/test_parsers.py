"""Tests for the parser modules."""

import pytest
from analyzer.parsers import PythonParser, FileParser
from analyzer.models.code_entities import EntityType, Visibility


class TestPythonParser:
    """Tests for PythonParser."""
    
    def test_parse_simple_function(self):
        """Test parsing a simple function."""
        code = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''
        parser = PythonParser()
        module = parser.parse_code(code)
        
        assert len(module.functions) == 1
        func = module.functions[0]
        assert func.name == "hello"
        assert len(func.parameters) == 1
        assert func.parameters[0].name == "name"
        assert func.parameters[0].type_annotation == "str"
        assert func.return_type == "str"
        assert func.docstring is not None
    
    def test_parse_class(self):
        """Test parsing a class."""
        code = '''
class Person:
    """A person class."""
    
    def __init__(self, name: str):
        self.name = name
    
    def greet(self) -> str:
        return f"Hello, I am {self.name}"
'''
        parser = PythonParser()
        module = parser.parse_code(code)
        
        assert len(module.classes) == 1
        cls = module.classes[0]
        assert cls.name == "Person"
        assert len(cls.methods) == 2
        assert cls.docstring is not None
    
    def test_parse_imports(self):
        """Test parsing import statements."""
        code = '''
import os
import sys as system
from typing import Optional, List
from . import local
from ..parent import something
'''
        parser = PythonParser()
        module = parser.parse_code(code)
        
        assert len(module.imports) == 6  # Note: Optional and List are separate imports
        
        # import os
        assert module.imports[0].module == "os"
        assert module.imports[0].alias is None
        
        # import sys as system
        assert module.imports[1].module == "sys"
        assert module.imports[1].alias == "system"
        
        # from typing import Optional
        assert module.imports[2].is_from_import
        assert module.imports[2].module == "typing"
        assert module.imports[2].name == "Optional"
    
    def test_parse_decorators(self):
        """Test parsing decorators."""
        code = '''
@staticmethod
def static_func():
    pass

@decorator(arg=True)
def decorated():
    pass
'''
        parser = PythonParser()
        module = parser.parse_code(code)
        
        assert len(module.functions) == 2
        assert len(module.functions[0].decorators) == 1
        assert module.functions[0].decorators[0].name == "staticmethod"
        
        assert module.functions[1].decorators[0].name == "decorator"
        assert "arg" in module.functions[1].decorators[0].keyword_arguments
    
    def test_parse_async_function(self):
        """Test parsing async functions."""
        code = '''
async def fetch_data(url: str):
    """Fetch data from URL."""
    return await client.get(url)
'''
        parser = PythonParser()
        module = parser.parse_code(code)
        
        assert len(module.functions) == 1
        func = module.functions[0]
        assert func.is_async
        assert func.entity_type == EntityType.ASYNC_FUNCTION
    
    def test_parse_visibility(self):
        """Test visibility detection."""
        code = '''
def public_func(): pass
def _protected_func(): pass
def __private_func(): pass
'''
        parser = PythonParser()
        module = parser.parse_code(code)
        
        assert module.functions[0].visibility == Visibility.PUBLIC
        assert module.functions[1].visibility == Visibility.PROTECTED
        assert module.functions[2].visibility == Visibility.PRIVATE


class TestFileParser:
    """Tests for FileParser."""
    
    def test_parse_file(self, temp_python_file):
        """Test parsing a file."""
        parser = FileParser()
        module = parser.parse_file(temp_python_file)
        
        assert module.name == "test_module"
        assert module.file_path == str(temp_python_file)
        assert len(module.functions) > 0
        assert len(module.classes) > 0
    
    def test_parse_directory(self, temp_project):
        """Test parsing a directory."""
        parser = FileParser()
        modules = parser.parse_directory(temp_project)
        
        assert len(modules) >= 3  # main.py, utils.py, module.py
        
        module_names = [m.name for m in modules]
        assert "main" in module_names
        assert "utils" in module_names
    
    def test_caching(self, temp_python_file):
        """Test file caching."""
        parser = FileParser()
        
        # Parse twice
        module1 = parser.parse_file(temp_python_file)
        module2 = parser.parse_file(temp_python_file)
        
        # Should get cached result
        cache_info = parser.get_cache_info()
        assert cache_info["size"] == 1
