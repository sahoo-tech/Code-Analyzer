"""
Anti-pattern detector.

Identifies problematic code patterns:
- God Class
- Spaghetti Code
- Feature Envy
- Data Class
- Long Parameter List
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from analyzer.models.code_entities import Module, Class, Function, Method
from analyzer.metrics.complexity import CyclomaticComplexityCalculator
from analyzer.logging_config import get_logger

logger = get_logger("patterns.anti_patterns")


class AntiPatternType(Enum):
    """Types of anti-patterns."""
    GOD_CLASS = "god_class"
    SPAGHETTI_CODE = "spaghetti_code"
    FEATURE_ENVY = "feature_envy"
    DATA_CLASS = "data_class"
    LONG_PARAMETER_LIST = "long_parameter_list"
    LONG_METHOD = "long_method"
    DEEP_NESTING = "deep_nesting"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    BLOB = "blob"


@dataclass
class AntiPatternMatch:
    """A detected anti-pattern."""
    pattern_type: AntiPatternType
    entity_name: str
    file_path: str
    line_number: int
    severity: str  # "low", "medium", "high"
    description: str
    suggestion: str
    metrics: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern_type.value,
            "entity": self.entity_name,
            "file": self.file_path,
            "line": self.line_number,
            "severity": self.severity,
            "description": self.description,
            "suggestion": self.suggestion,
            "metrics": self.metrics,
        }


class AntiPatternDetector:
    """Detects anti-patterns in code."""
    
    # Thresholds
    GOD_CLASS_METHOD_THRESHOLD = 20
    GOD_CLASS_LOC_THRESHOLD = 500
    LONG_METHOD_THRESHOLD = 50
    LONG_PARAM_THRESHOLD = 5
    DEEP_NESTING_THRESHOLD = 4
    DATA_CLASS_METHOD_RATIO = 0.8  # 80% getters/setters
    
    def detect(self, modules: list[Module]) -> list[AntiPatternMatch]:
        """
        Detect anti-patterns in modules.
        
        Args:
            modules: List of parsed modules
            
        Returns:
            List of detected anti-patterns
        """
        patterns = []
        
        for module in modules:
            # Check classes
            for cls in module.classes:
                patterns.extend(self._analyze_class(cls, module.file_path))
            
            # Check functions
            for func in module.functions:
                patterns.extend(self._analyze_function(func, module.file_path))
        
        return patterns
    
    def _analyze_class(self, cls: Class, file_path: str) -> list[AntiPatternMatch]:
        """Analyze a class for anti-patterns."""
        patterns = []
        
        # God Class
        god_class = self._detect_god_class(cls, file_path)
        if god_class:
            patterns.append(god_class)
        
        # Data Class
        data_class = self._detect_data_class(cls, file_path)
        if data_class:
            patterns.append(data_class)
        
        # Check methods within class
        for method in cls.methods:
            patterns.extend(self._analyze_function(method, file_path, cls.name))
        
        return patterns
    
    def _analyze_function(
        self, 
        func: Function, 
        file_path: str,
        class_name: Optional[str] = None
    ) -> list[AntiPatternMatch]:
        """Analyze a function for anti-patterns."""
        patterns = []
        entity_name = f"{class_name}.{func.name}" if class_name else func.name
        
        # Long Method
        long_method = self._detect_long_method(func, file_path, entity_name)
        if long_method:
            patterns.append(long_method)
        
        # Long Parameter List
        long_params = self._detect_long_parameter_list(func, file_path, entity_name)
        if long_params:
            patterns.append(long_params)
        
        return patterns
    
    def _detect_god_class(self, cls: Class, file_path: str) -> Optional[AntiPatternMatch]:
        """Detect God Class anti-pattern."""
        method_count = len(cls.methods)
        loc = cls.location.line_count
        var_count = len(cls.class_variables) + len(cls.instance_variables)
        
        is_god_class = (
            method_count > self.GOD_CLASS_METHOD_THRESHOLD or
            loc > self.GOD_CLASS_LOC_THRESHOLD
        )
        
        if not is_god_class:
            return None
        
        # Determine severity
        if method_count > self.GOD_CLASS_METHOD_THRESHOLD * 2 or loc > self.GOD_CLASS_LOC_THRESHOLD * 2:
            severity = "high"
        elif method_count > self.GOD_CLASS_METHOD_THRESHOLD * 1.5 or loc > self.GOD_CLASS_LOC_THRESHOLD * 1.5:
            severity = "medium"
        else:
            severity = "low"
        
        return AntiPatternMatch(
            pattern_type=AntiPatternType.GOD_CLASS,
            entity_name=cls.name,
            file_path=file_path,
            line_number=cls.location.start_line,
            severity=severity,
            description=f"Class has too many responsibilities ({method_count} methods, {loc} lines)",
            suggestion="Split into smaller, focused classes with single responsibility",
            metrics={
                "methods": method_count,
                "lines": loc,
                "variables": var_count,
            }
        )
    
    def _detect_data_class(self, cls: Class, file_path: str) -> Optional[AntiPatternMatch]:
        """Detect Data Class anti-pattern (class with only getters/setters)."""
        if cls.is_dataclass:
            return None  # Intentional data classes are fine
        
        total_methods = len(cls.methods)
        if total_methods < 3:
            return None  # Too few methods to determine
        
        # Count getter/setter-like methods
        accessor_methods = 0
        for method in cls.methods:
            name = method.name
            if name.startswith(('get_', 'set_', 'is_', 'has_')):
                accessor_methods += 1
            elif name.startswith('_') and name not in ('__init__', '__repr__', '__str__'):
                continue  # Skip private methods
        
        # Exclude dunder methods from ratio
        non_dunder = [m for m in cls.methods if not m.name.startswith('__')]
        if not non_dunder:
            return None
        
        ratio = accessor_methods / len(non_dunder)
        
        if ratio < self.DATA_CLASS_METHOD_RATIO:
            return None
        
        return AntiPatternMatch(
            pattern_type=AntiPatternType.DATA_CLASS,
            entity_name=cls.name,
            file_path=file_path,
            line_number=cls.location.start_line,
            severity="low",
            description=f"Class contains mostly accessor methods ({ratio:.0%})",
            suggestion="Consider using @dataclass decorator or moving logic into this class",
            metrics={
                "accessor_ratio": ratio,
                "total_methods": total_methods,
            }
        )
    
    def _detect_long_method(
        self, 
        func: Function, 
        file_path: str,
        entity_name: str
    ) -> Optional[AntiPatternMatch]:
        """Detect Long Method anti-pattern."""
        loc = func.location.line_count
        
        if loc <= self.LONG_METHOD_THRESHOLD:
            return None
        
        # Determine severity
        if loc > self.LONG_METHOD_THRESHOLD * 3:
            severity = "high"
        elif loc > self.LONG_METHOD_THRESHOLD * 2:
            severity = "medium"
        else:
            severity = "low"
        
        return AntiPatternMatch(
            pattern_type=AntiPatternType.LONG_METHOD,
            entity_name=entity_name,
            file_path=file_path,
            line_number=func.location.start_line,
            severity=severity,
            description=f"Method is too long ({loc} lines)",
            suggestion="Break into smaller, focused helper methods",
            metrics={"lines": loc}
        )
    
    def _detect_long_parameter_list(
        self, 
        func: Function, 
        file_path: str,
        entity_name: str
    ) -> Optional[AntiPatternMatch]:
        """Detect Long Parameter List anti-pattern."""
        # Filter out self, cls, *args, **kwargs
        params = [
            p for p in func.parameters
            if p.name not in ('self', 'cls') and p.kind not in ('var_positional', 'var_keyword')
        ]
        
        param_count = len(params)
        
        if param_count <= self.LONG_PARAM_THRESHOLD:
            return None
        
        # Determine severity
        if param_count > self.LONG_PARAM_THRESHOLD * 2:
            severity = "high"
        elif param_count > self.LONG_PARAM_THRESHOLD * 1.5:
            severity = "medium"
        else:
            severity = "low"
        
        return AntiPatternMatch(
            pattern_type=AntiPatternType.LONG_PARAMETER_LIST,
            entity_name=entity_name,
            file_path=file_path,
            line_number=func.location.start_line,
            severity=severity,
            description=f"Too many parameters ({param_count})",
            suggestion="Use parameter objects, builder pattern, or split into multiple methods",
            metrics={
                "param_count": param_count,
                "params": [p.name for p in params],
            }
        )


def detect_anti_patterns(modules: list[Module]) -> list[AntiPatternMatch]:
    """
    Detect anti-patterns in modules.
    
    Args:
        modules: List of parsed modules
        
    Returns:
        List of detected anti-patterns
    """
    detector = AntiPatternDetector()
    return detector.detect(modules)
