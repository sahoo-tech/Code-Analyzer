"""
Import analyzer.

Analyzes import statements to understand module dependencies.
"""

import ast
import sys
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field

from analyzer.models.code_entities import Import, Module
from analyzer.config import get_config
from analyzer.logging_config import get_logger

logger = get_logger("dependencies.imports")


@dataclass
class ImportInfo:
    """Extended import information with resolution details."""
    import_obj: Import
    resolved_path: Optional[str] = None
    is_stdlib: bool = False
    is_third_party: bool = False
    is_local: bool = False
    is_resolvable: bool = True
    package: Optional[str] = None


@dataclass  
class ImportAnalysisResult:
    """Result of import analysis."""
    imports: list[ImportInfo] = field(default_factory=list)
    stdlib_imports: list[str] = field(default_factory=list)
    third_party_imports: list[str] = field(default_factory=list)
    local_imports: list[str] = field(default_factory=list)
    unresolved_imports: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "total": len(self.imports),
            "stdlib": self.stdlib_imports,
            "third_party": self.third_party_imports,
            "local": self.local_imports,
            "unresolved": self.unresolved_imports,
        }


class ImportAnalyzer:
    """
    Analyzes imports in Python code.
    
    Resolves import paths and categorizes them as:
    - Standard library
    - Third-party packages
    - Local/project imports
    """
    
    # Common stdlib module names (subset for quick checks)
    STDLIB_MODULES = {
        'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections',
        'concurrent', 'configparser', 'contextlib', 'copy', 'csv',
        'dataclasses', 'datetime', 'decimal', 'difflib', 'email',
        'enum', 'fileinput', 'fnmatch', 'fractions', 'functools',
        'gc', 'getpass', 'glob', 'gzip', 'hashlib', 'heapq', 'hmac',
        'html', 'http', 'importlib', 'inspect', 'io', 'itertools',
        'json', 'keyword', 'logging', 'math', 'mimetypes', 'multiprocessing',
        'numbers', 'operator', 'os', 'pathlib', 'pickle', 'platform',
        'pprint', 'queue', 'random', 're', 'shutil', 'signal', 'socket',
        'sqlite3', 'ssl', 'statistics', 'string', 'struct', 'subprocess',
        'sys', 'tarfile', 'tempfile', 'textwrap', 'threading', 'time',
        'timeit', 'tokenize', 'traceback', 'types', 'typing', 'unicodedata',
        'unittest', 'urllib', 'uuid', 'warnings', 'weakref', 'xml',
        'zipfile', 'zlib',
    }
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = Path(project_root) if project_root else None
        self.config = get_config()
        self._stdlib_modules: Optional[set[str]] = None
    
    @property
    def stdlib_modules(self) -> set[str]:
        """Get all stdlib module names."""
        if self._stdlib_modules is None:
            self._stdlib_modules = self._get_stdlib_modules()
        return self._stdlib_modules
    
    def _get_stdlib_modules(self) -> set[str]:
        """Dynamically get stdlib module names."""
        stdlib = set(self.STDLIB_MODULES)
        
        # Try to get from sys.stdlib_module_names (Python 3.10+)
        if hasattr(sys, 'stdlib_module_names'):
            stdlib.update(sys.stdlib_module_names)
        
        return stdlib
    
    def analyze(self, module: Module) -> ImportAnalysisResult:
        """
        Analyze imports in a module.
        
        Args:
            module: Parsed module to analyze
            
        Returns:
            ImportAnalysisResult with categorized imports
        """
        result = ImportAnalysisResult()
        
        for imp in module.imports:
            info = self._analyze_import(imp, module.file_path)
            result.imports.append(info)
            
            # Categorize
            module_name = imp.module.split('.')[0]
            
            if info.is_stdlib:
                if module_name not in result.stdlib_imports:
                    result.stdlib_imports.append(module_name)
            elif info.is_third_party:
                if module_name not in result.third_party_imports:
                    result.third_party_imports.append(module_name)
            elif info.is_local:
                full_name = imp.full_name
                if full_name not in result.local_imports:
                    result.local_imports.append(full_name)
            
            if not info.is_resolvable:
                result.unresolved_imports.append(imp.full_name)
        
        return result
    
    def _analyze_import(self, imp: Import, source_file: str) -> ImportInfo:
        """Analyze a single import."""
        info = ImportInfo(import_obj=imp)
        
        # Get base module name
        base_module = imp.module.split('.')[0] if imp.module else ''
        
        # Check if stdlib
        if base_module in self.stdlib_modules:
            info.is_stdlib = True
            info.is_resolvable = True
            return info
        
        # Check if relative import
        if imp.is_relative:
            info.is_local = True
            info.resolved_path = self._resolve_relative_import(imp, source_file)
            info.is_resolvable = info.resolved_path is not None
            return info
        
        # Try to resolve as local import
        if self.project_root:
            resolved = self._resolve_local_import(imp)
            if resolved:
                info.is_local = True
                info.resolved_path = resolved
                info.is_resolvable = True
                return info
        
        # Assume third-party
        info.is_third_party = True
        info.is_resolvable = self._is_installed(base_module)
        
        return info
    
    def _resolve_relative_import(self, imp: Import, source_file: str) -> Optional[str]:
        """Resolve a relative import to absolute path."""
        if not source_file:
            return None
        
        source_path = Path(source_file)
        
        # Go up 'level' directories
        current = source_path.parent
        for _ in range(imp.level - 1):
            current = current.parent
        
        # Resolve the module path
        if imp.module:
            parts = imp.module.split('.')
            for part in parts:
                current = current / part
        
        # Check as package or module
        if (current / '__init__.py').exists():
            return str(current / '__init__.py')
        
        module_file = current.with_suffix('.py')
        if module_file.exists():
            return str(module_file)
        
        return None
    
    def _resolve_local_import(self, imp: Import) -> Optional[str]:
        """Try to resolve import as local project import."""
        if not self.project_root:
            return None
        
        parts = imp.module.split('.')
        current = self.project_root
        
        for part in parts:
            current = current / part
        
        # Check as package
        if (current / '__init__.py').exists():
            return str(current / '__init__.py')
        
        # Check as module
        module_file = current.with_suffix('.py')
        if module_file.exists():
            return str(module_file)
        
        return None
    
    def _is_installed(self, module_name: str) -> bool:
        """Check if a module is installed."""
        try:
            import importlib.util
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ModuleNotFoundError, ValueError):
            return False


def analyze_imports(
    module: Module,
    project_root: Optional[Path] = None
) -> ImportAnalysisResult:
    """
    Analyze imports in a module.
    
    Args:
        module: Parsed module to analyze
        project_root: Root directory of the project
        
    Returns:
        ImportAnalysisResult with categorized imports
    """
    analyzer = ImportAnalyzer(project_root)
    return analyzer.analyze(module)
