"""Parsers module initialization."""

from analyzer.parsers.base import BaseParser
from analyzer.parsers.python_parser import PythonParser
from analyzer.parsers.file_parser import FileParser

__all__ = [
    "BaseParser",
    "PythonParser",
    "FileParser",
]
