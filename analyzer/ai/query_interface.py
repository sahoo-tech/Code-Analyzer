"""
Natural language query interface.

Allows querying code using natural language patterns:
- Find functions that do X
- Show classes inheriting from Y
- List all async methods
"""

import re
from dataclasses import dataclass
from typing import Optional, Any
from enum import Enum

from analyzer.models.code_entities import Module, Class, Function, Method
from analyzer.logging_config import get_logger

logger = get_logger("ai.query")


class QueryType(Enum):
    """Types of code queries."""
    FIND_FUNCTION = "find_function"
    FIND_CLASS = "find_class"
    FIND_METHOD = "find_method"
    FIND_IMPORT = "find_import"
    LIST_ALL = "list_all"
    SEARCH = "search"


@dataclass
class QueryResult:
    """Result of a code query."""
    query: str
    query_type: QueryType
    results: list[Any]
    count: int
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "type": self.query_type.value,
            "count": self.count,
            "results": [
                r.to_dict() if hasattr(r, 'to_dict') else str(r)
                for r in self.results
            ],
        }


class QueryInterface:
    """Natural language interface for querying code."""
    
    def __init__(self, modules: list[Module]):
        self.modules = modules
        self._build_index()
    
    def _build_index(self) -> None:
        """Build search indices for faster querying."""
        self._functions: list[tuple[str, Function, Module]] = []
        self._classes: list[tuple[str, Class, Module]] = []
        self._methods: list[tuple[str, Method, Class, Module]] = []
        
        for module in self.modules:
            for func in module.functions:
                self._functions.append((func.name.lower(), func, module))
            
            for cls in module.classes:
                self._classes.append((cls.name.lower(), cls, module))
                
                for method in cls.methods:
                    self._methods.append((method.name.lower(), method, cls, module))
    
    def query(self, query_string: str) -> QueryResult:
        """
        Execute a natural language query.
        
        Args:
            query_string: Natural language query
            
        Returns:
            QueryResult with matching items
        """
        query_lower = query_string.lower().strip()
        
        # Pattern matching for common queries
        patterns = [
            (r"find (?:all )?functions? (?:named |called )?['\"]?(\w+)['\"]?", self._find_function),
            (r"find (?:all )?class(?:es)? (?:named |called )?['\"]?(\w+)['\"]?", self._find_class),
            (r"find (?:all )?methods? (?:named |called )?['\"]?(\w+)['\"]?", self._find_method),
            (r"list (?:all )?async (?:functions?|methods?)", self._list_async),
            (r"list (?:all )?abstract (?:methods?|classes?)", self._list_abstract),
            (r"list (?:all )?classes? (?:that )?inherit(?:ing)? (?:from )?['\"]?(\w+)['\"]?", self._find_subclasses),
            (r"show (?:all )?imports? (?:of |from )?['\"]?(\w+)['\"]?", self._find_imports),
            (r"search ['\"]?(.+)['\"]?", self._search),
        ]
        
        for pattern, handler in patterns:
            match = re.match(pattern, query_lower)
            if match:
                return handler(query_string, match)
        
        # Default to search
        return self._search(query_string, None)
    
    def _find_function(self, query: str, match: re.Match) -> QueryResult:
        """Find functions by name."""
        name = match.group(1).lower()
        results = []
        
        for func_name, func, module in self._functions:
            if name in func_name or func_name in name:
                results.append({
                    "type": "function",
                    "name": func.name,
                    "module": module.name,
                    "file": module.file_path,
                    "line": func.location.start_line,
                    "signature": func.signature,
                })
        
        return QueryResult(
            query=query,
            query_type=QueryType.FIND_FUNCTION,
            results=results,
            count=len(results),
        )
    
    def _find_class(self, query: str, match: re.Match) -> QueryResult:
        """Find classes by name."""
        name = match.group(1).lower()
        results = []
        
        for class_name, cls, module in self._classes:
            if name in class_name or class_name in name:
                results.append({
                    "type": "class",
                    "name": cls.name,
                    "module": module.name,
                    "file": module.file_path,
                    "line": cls.location.start_line,
                    "bases": cls.bases,
                    "methods": len(cls.methods),
                })
        
        return QueryResult(
            query=query,
            query_type=QueryType.FIND_CLASS,
            results=results,
            count=len(results),
        )
    
    def _find_method(self, query: str, match: re.Match) -> QueryResult:
        """Find methods by name."""
        name = match.group(1).lower()
        results = []
        
        for method_name, method, cls, module in self._methods:
            if name in method_name or method_name in name:
                results.append({
                    "type": "method",
                    "name": method.name,
                    "class": cls.name,
                    "module": module.name,
                    "file": module.file_path,
                    "line": method.location.start_line,
                    "is_static": method.is_static,
                    "is_async": method.is_async,
                })
        
        return QueryResult(
            query=query,
            query_type=QueryType.FIND_METHOD,
            results=results,
            count=len(results),
        )
    
    def _list_async(self, query: str, match: re.Match) -> QueryResult:
        """List all async functions and methods."""
        results = []
        
        # Async functions
        for _, func, module in self._functions:
            if func.is_async:
                results.append({
                    "type": "function",
                    "name": func.name,
                    "module": module.name,
                    "file": module.file_path,
                })
        
        # Async methods
        for _, method, cls, module in self._methods:
            if method.is_async:
                results.append({
                    "type": "method",
                    "name": f"{cls.name}.{method.name}",
                    "module": module.name,
                    "file": module.file_path,
                })
        
        return QueryResult(
            query=query,
            query_type=QueryType.LIST_ALL,
            results=results,
            count=len(results),
        )
    
    def _list_abstract(self, query: str, match: re.Match) -> QueryResult:
        """List abstract classes and methods."""
        results = []
        
        # Abstract classes
        for _, cls, module in self._classes:
            if cls.is_abstract:
                results.append({
                    "type": "abstract_class",
                    "name": cls.name,
                    "module": module.name,
                    "file": module.file_path,
                })
        
        # Abstract methods
        for _, method, cls, module in self._methods:
            if method.is_abstract:
                results.append({
                    "type": "abstract_method",
                    "name": f"{cls.name}.{method.name}",
                    "module": module.name,
                    "file": module.file_path,
                })
        
        return QueryResult(
            query=query,
            query_type=QueryType.LIST_ALL,
            results=results,
            count=len(results),
        )
    
    def _find_subclasses(self, query: str, match: re.Match) -> QueryResult:
        """Find classes inheriting from a base class."""
        base_name = match.group(1)
        results = []
        
        for _, cls, module in self._classes:
            if base_name in cls.bases:
                results.append({
                    "type": "class",
                    "name": cls.name,
                    "module": module.name,
                    "file": module.file_path,
                    "bases": cls.bases,
                })
        
        return QueryResult(
            query=query,
            query_type=QueryType.FIND_CLASS,
            results=results,
            count=len(results),
        )
    
    def _find_imports(self, query: str, match: re.Match) -> QueryResult:
        """Find imports of a module."""
        module_name = match.group(1).lower()
        results = []
        
        for module in self.modules:
            for imp in module.imports:
                if module_name in imp.module.lower():
                    results.append({
                        "type": "import",
                        "module": imp.module,
                        "name": imp.name,
                        "alias": imp.alias,
                        "used_in": module.name,
                        "file": module.file_path,
                    })
        
        return QueryResult(
            query=query,
            query_type=QueryType.FIND_IMPORT,
            results=results,
            count=len(results),
        )
    
    def _search(self, query: str, match: Optional[re.Match]) -> QueryResult:
        """General search across all entities."""
        search_term = (match.group(1) if match else query).lower()
        results = []
        
        # Search functions
        for func_name, func, module in self._functions:
            if search_term in func_name:
                results.append({
                    "type": "function",
                    "name": func.name,
                    "module": module.name,
                })
        
        # Search classes
        for class_name, cls, module in self._classes:
            if search_term in class_name:
                results.append({
                    "type": "class",
                    "name": cls.name,
                    "module": module.name,
                })
        
        # Search methods
        for method_name, method, cls, module in self._methods:
            if search_term in method_name:
                results.append({
                    "type": "method",
                    "name": f"{cls.name}.{method.name}",
                    "module": module.name,
                })
        
        return QueryResult(
            query=query,
            query_type=QueryType.SEARCH,
            results=results,
            count=len(results),
        )


def query_codebase(modules: list[Module], query: str) -> QueryResult:
    """
    Query a codebase using natural language.
    
    Args:
        modules: List of parsed modules
        query: Natural language query
        
    Returns:
        QueryResult with matching items
    """
    interface = QueryInterface(modules)
    return interface.query(query)
