"""
Abstract base parser.

Defines the interface that all language parsers must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from analyzer.models.code_entities import Module


class BaseParser(ABC):
    """Abstract base class for language parsers."""
    
    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions this parser supports."""
        pass
    
    @property
    @abstractmethod
    def language(self) -> str:
        """Language this parser handles."""
        pass
    
    @abstractmethod
    def parse_file(self, path: Union[str, Path]) -> Module:
        """
        Parse a source file.
        
        Args:
            path: Path to the source file
            
        Returns:
            Parsed Module object
        """
        pass
    
    @abstractmethod
    def parse_code(self, code: str, filename: str = "<string>") -> Module:
        """
        Parse source code string.
        
        Args:
            code: Source code string
            filename: Optional filename for error messages
            
        Returns:
            Parsed Module object
        """
        pass
    
    def can_parse(self, path: Union[str, Path]) -> bool:
        """Check if this parser can handle the given file."""
        path = Path(path)
        return path.suffix in self.supported_extensions
