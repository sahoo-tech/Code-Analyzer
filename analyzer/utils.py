"""
Utility functions for the Code Analyzer.

Provides shared utilities for:
- File handling
- Caching
- Decorators
- String manipulation
- Path handling
"""

import os
import re
import hashlib
import functools
import time
from pathlib import Path
from typing import (
    Optional, Any, Callable, TypeVar, Union, Iterator, 
    Sequence, Hashable
)
from fnmatch import fnmatch
from concurrent.futures import ThreadPoolExecutor, as_completed

from analyzer.exceptions import FileSystemError, EncodingError
from analyzer.logging_config import get_logger

logger = get_logger("utils")

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# ============================================================================
# File Utilities
# ============================================================================

def read_file(
    path: Union[str, Path],
    encoding: str = "utf-8",
    fallback_encodings: Optional[list[str]] = None
) -> str:
    """
    Read file contents with encoding fallback.
    
    Args:
        path: Path to the file
        encoding: Primary encoding to try
        fallback_encodings: List of fallback encodings
        
    Returns:
        File contents as string
        
    Raises:
        FileSystemError: If file cannot be read
        EncodingError: If file cannot be decoded
    """
    path = Path(path)
    fallback_encodings = fallback_encodings or ["latin-1", "cp1252"]
    
    if not path.exists():
        raise FileSystemError(f"File not found: {path}")
    
    if not path.is_file():
        raise FileSystemError(f"Not a file: {path}")
    
    encodings_to_try = [encoding] + fallback_encodings
    
    for enc in encodings_to_try:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except PermissionError as e:
            raise FileSystemError(f"Permission denied: {path}") from e
        except OSError as e:
            raise FileSystemError(f"Error reading file {path}: {e}") from e
    
    raise EncodingError(
        f"Could not decode file with any encoding: {path}",
        {"tried_encodings": encodings_to_try}
    )


def get_file_hash(path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    Calculate hash of a file's contents.
    
    Args:
        path: Path to the file
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hexadecimal hash string
    """
    path = Path(path)
    hasher = hashlib.new(algorithm)
    
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def find_files(
    directory: Union[str, Path],
    include_patterns: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
    follow_symlinks: bool = False,
    max_depth: Optional[int] = None,
) -> Iterator[Path]:
    """
    Find files matching patterns in a directory.
    
    Args:
        directory: Root directory to search
        include_patterns: Glob patterns to include (default: ["*.py"])
        exclude_patterns: Glob patterns to exclude
        follow_symlinks: Whether to follow symbolic links
        max_depth: Maximum directory depth to search
        
    Yields:
        Path objects for matching files
    """
    directory = Path(directory)
    include_patterns = include_patterns or ["*.py"]
    exclude_patterns = exclude_patterns or []
    
    if not directory.exists():
        raise FileSystemError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise FileSystemError(f"Not a directory: {directory}")
    
    def should_include(path: Path) -> bool:
        relative = path.relative_to(directory)
        str_path = str(relative)
        
        # Check exclusions first
        for pattern in exclude_patterns:
            if fnmatch(str_path, pattern) or fnmatch(path.name, pattern):
                return False
        
        # Check inclusions
        for pattern in include_patterns:
            if fnmatch(path.name, pattern):
                return True
        
        return False
    
    def walk_dir(dir_path: Path, depth: int = 0) -> Iterator[Path]:
        if max_depth is not None and depth > max_depth:
            return
        
        try:
            entries = list(dir_path.iterdir())
        except PermissionError:
            logger.warning(f"Permission denied: {dir_path}")
            return
        
        for entry in entries:
            if entry.is_symlink() and not follow_symlinks:
                continue
            
            if entry.is_file():
                if should_include(entry):
                    yield entry
            elif entry.is_dir():
                # Skip hidden directories
                if entry.name.startswith("."):
                    continue
                yield from walk_dir(entry, depth + 1)
    
    yield from walk_dir(directory)


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> str:
    """Get path relative to base, or absolute if not relative."""
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return str(path)


# ============================================================================
# Caching Utilities
# ============================================================================

def memoize(func: F) -> F:
    """Simple memoization decorator for functions with hashable arguments."""
    cache: dict[tuple, Any] = {}
    
    @functools.wraps(func)
    def wrapper(*args: Hashable, **kwargs: Any) -> Any:
        # Create cache key from arguments
        key = (args, tuple(sorted(kwargs.items())))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    wrapper.cache_clear = lambda: cache.clear()  # type: ignore
    return wrapper  # type: ignore


def timed_cache(ttl_seconds: int = 300) -> Callable[[F], F]:
    """
    Decorator that caches results with a time-to-live.
    
    Args:
        ttl_seconds: Cache lifetime in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        cache: dict[tuple, tuple[float, Any]] = {}
        
        @functools.wraps(func)
        def wrapper(*args: Hashable, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            current_time = time.time()
            
            if key in cache:
                cached_time, cached_result = cache[key]
                if current_time - cached_time < ttl_seconds:
                    return cached_result
            
            result = func(*args, **kwargs)
            cache[key] = (current_time, result)
            return result
        
        wrapper.cache_clear = lambda: cache.clear()  # type: ignore
        return wrapper  # type: ignore
    
    return decorator


# ============================================================================
# String Utilities
# ============================================================================

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text (collapse multiple spaces, strip)."""
    return " ".join(text.split())


def count_lines(text: str) -> dict[str, int]:
    """
    Count different types of lines in text.
    
    Returns:
        Dictionary with total, blank, and non_blank counts
    """
    lines = text.splitlines()
    blank = sum(1 for line in lines if not line.strip())
    
    return {
        "total": len(lines),
        "blank": blank,
        "non_blank": len(lines) - blank,
    }


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate string to max length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_identifiers(text: str) -> list[str]:
    """Extract Python identifiers from text."""
    pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
    return re.findall(pattern, text)


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def to_camel_case(name: str) -> str:
    """Convert snake_case to CamelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def to_pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(x.title() for x in name.split("_"))


# ============================================================================
# Parallel Processing
# ============================================================================

def parallel_map(
    func: Callable[[T], Any],
    items: Sequence[T],
    max_workers: Optional[int] = None,
    desc: Optional[str] = None,
) -> list[Any]:
    """
    Apply function to items in parallel.
    
    Args:
        func: Function to apply
        items: Items to process
        max_workers: Maximum number of threads
        desc: Description for progress logging
        
    Returns:
        List of results in order
    """
    if not items:
        return []
    
    max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
    results: list[Any] = [None] * len(items)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(func, item): i 
            for i, item in enumerate(items)
        }
        
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                logger.warning(f"Error processing item {index}: {e}")
                results[index] = None
            
            completed += 1
            if desc and completed % 10 == 0:
                logger.debug(f"{desc}: {completed}/{len(items)}")
    
    return results


# ============================================================================
# Timing Utilities  
# ============================================================================

def timer(name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to time function execution.
    
    Args:
        name: Optional name for the timer (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        timer_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.debug(f"{timer_name} completed in {elapsed:.3f}s")
        
        return wrapper  # type: ignore
    
    return decorator


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: float = 0
        self.elapsed: float = 0
    
    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        logger.debug(f"{self.name} completed in {self.elapsed:.3f}s")


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_path(path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate and normalize a path.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        
    Returns:
        Normalized Path object
        
    Raises:
        FileSystemError: If validation fails
    """
    path = Path(path).resolve()
    
    if must_exist and not path.exists():
        raise FileSystemError(f"Path does not exist: {path}")
    
    return path


def is_python_file(path: Union[str, Path]) -> bool:
    """Check if path is a Python file."""
    path = Path(path)
    return path.is_file() and path.suffix == ".py"


def is_test_file(path: Union[str, Path]) -> bool:
    """Check if path is a test file."""
    path = Path(path)
    name = path.name.lower()
    return (
        name.startswith("test_") or 
        name.endswith("_test.py") or  
        "tests" in path.parts
    )
