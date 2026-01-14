

import os
import hashlib
from pathlib import Path
from typing import Optional, Union, Iterator, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from analyzer.parsers.base import BaseParser
from analyzer.parsers.python_parser import PythonParser
from analyzer.models.code_entities import Module
from analyzer.config import get_config, AnalyzerConfig
from analyzer.exceptions import FileSystemError, ParsingError
from analyzer.utils import find_files, get_file_hash
from analyzer.logging_config import get_logger

logger = get_logger("parsers.file")


class FileParser:

    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or get_config()
        self._parsers: dict[str, BaseParser] = {}
        self._cache: dict[str, tuple[str, Module]] = {}  # path -> (hash, module)
        
        # Register default parsers
        self.register_parser(PythonParser())
    
    def register_parser(self, parser: BaseParser) -> None:
        """Register a language parser."""
        for ext in parser.supported_extensions:
            self._parsers[ext] = parser
            logger.debug(f"Registered parser for {ext}: {parser.__class__.__name__}")
    
    def get_parser(self, path: Union[str, Path]) -> Optional[BaseParser]:
        """Get appropriate parser for a file."""
        path = Path(path)
        return self._parsers.get(path.suffix)
    
    def parse_file(
        self, 
        path: Union[str, Path],
        use_cache: bool = True
    ) -> Module:
   
        path = Path(path).resolve()
        str_path = str(path)
        
        # Check cache
        if use_cache and self.config.cache.enabled:
            file_hash = get_file_hash(path)
            
            if str_path in self._cache:
                cached_hash, cached_module = self._cache[str_path]
                if cached_hash == file_hash:
                    logger.debug(f"Cache hit for {path}")
                    return cached_module
        
        # Get parser
        parser = self.get_parser(path)
        if parser is None:
            raise ParsingError(f"No parser available for {path.suffix}")
        
        # Parse
        logger.debug(f"Parsing {path}")
        module = parser.parse_file(path)
        
        # Cache result
        if use_cache and self.config.cache.enabled:
            file_hash = get_file_hash(path)
            self._cache[str_path] = (file_hash, module)
        
        return module
    
    def parse_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[Module]:
     
   
        directory = Path(directory).resolve()
        
        if not directory.exists():
            raise FileSystemError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise FileSystemError(f"Not a directory: {directory}")
        
        # Use config defaults if not specified
        include_patterns = include_patterns or self.config.include_patterns
        exclude_patterns = exclude_patterns or self.config.exclude_patterns
        max_workers = max_workers or self.config.max_workers
        
        # Find all matching files
        files = list(find_files(
            directory,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            follow_symlinks=self.config.follow_symlinks,
            max_depth=None if recursive else 0,
        ))
        
        logger.info(f"Found {len(files)} files to parse in {directory}")
        
        if not files:
            return []
        
        # Parse files
        modules: list[Optional[Module]] = [None] * len(files)
        errors: list[tuple[Path, Exception]] = []
        
        def parse_one(index: int, path: Path) -> tuple[int, Optional[Module], Optional[Exception]]:
            try:
                module = self.parse_file(path)
                return (index, module, None)
            except Exception as e:
                return (index, None, e)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(parse_one, i, path): i
                for i, path in enumerate(files)
            }
            
            completed = 0
            for future in as_completed(futures):
                index, module, error = future.result()
                
                if error:
                    errors.append((files[index], error))
                    logger.warning(f"Failed to parse {files[index]}: {error}")
                else:
                    modules[index] = module
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(files))
        
        # Filter out None values
        result = [m for m in modules if m is not None]
        
        if errors:
            logger.warning(f"Failed to parse {len(errors)} files")
        
        logger.info(f"Successfully parsed {len(result)} files")
        return result
    
    def parse_code(
        self,
        code: str,
        language: str = "python",
        filename: str = "<string>"
    ) -> Module:
        """
        Parse code string.
        
        Args:
            code: Source code string
            language: Language of the code
            filename: Optional filename for error messages
            
        Returns:
            Parsed Module object
        """
        # Map language to extension
        ext_map = {
            "python": ".py",
            "py": ".py",
        }
        
        ext = ext_map.get(language.lower())
        if ext is None:
            raise ParsingError(f"Unsupported language: {language}")
        
        parser = self._parsers.get(ext)
        if parser is None:
            raise ParsingError(f"No parser for language: {language}")
        
        return parser.parse_code(code, filename)
    
    def iter_files(
        self,
        directory: Union[str, Path],
        include_patterns: Optional[list[str]] = None,
        exclude_patterns: Optional[list[str]] = None,
    ) -> Iterator[Path]:
        """
        Iterate over files in directory that can be parsed.
        
        Args:
            directory: Root directory
            include_patterns: Glob patterns to include
            exclude_patterns: Glob patterns to exclude
            
        Yields:
            Path objects for parseable files
        """
        include_patterns = include_patterns or self.config.include_patterns
        exclude_patterns = exclude_patterns or self.config.exclude_patterns
        
        yield from find_files(
            directory,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            follow_symlinks=self.config.follow_symlinks,
        )
    
    def clear_cache(self) -> None:
        """Clear the parse cache."""
        self._cache.clear()
        logger.debug("Parse cache cleared")
    
    def get_cache_info(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "files": list(self._cache.keys()),
        }
