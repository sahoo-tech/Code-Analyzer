"""
Design pattern detector.

Identifies common design patterns in code:
- Singleton
- Factory
- Observer
- Decorator
- Strategy
- Builder
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from analyzer.models.code_entities import Module, Class, Method
from analyzer.logging_config import get_logger

logger = get_logger("patterns.design_patterns")


class PatternType(Enum):
    """Types of design patterns."""
    SINGLETON = "singleton"
    FACTORY = "factory"
    ABSTRACT_FACTORY = "abstract_factory"
    BUILDER = "builder"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    DECORATOR = "decorator"
    ADAPTER = "adapter"
    FACADE = "facade"
    TEMPLATE_METHOD = "template_method"
    ITERATOR = "iterator"
    COMMAND = "command"


@dataclass
class PatternMatch:
    """A detected design pattern."""
    pattern_type: PatternType
    class_name: str
    file_path: str
    line_number: int
    confidence: float  # 0.0 to 1.0
    evidence: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern_type.value,
            "class": self.class_name,
            "file": self.file_path,
            "line": self.line_number,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


class DesignPatternDetector:
    """Detects design patterns in code."""
    
    def detect(self, modules: list[Module]) -> list[PatternMatch]:
        """
        Detect design patterns in modules.
        
        Args:
            modules: List of parsed modules
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        for module in modules:
            for cls in module.classes:
                patterns.extend(self._analyze_class(cls, module.file_path))
        
        return patterns
    
    def _analyze_class(self, cls: Class, file_path: str) -> list[PatternMatch]:
        """Analyze a class for design patterns."""
        patterns = []
        
        # Singleton detection
        singleton = self._detect_singleton(cls, file_path)
        if singleton:
            patterns.append(singleton)
        
        # Factory detection
        factory = self._detect_factory(cls, file_path)
        if factory:
            patterns.append(factory)
        
        # Observer detection
        observer = self._detect_observer(cls, file_path)
        if observer:
            patterns.append(observer)
        
        # Decorator detection
        decorator = self._detect_decorator(cls, file_path)
        if decorator:
            patterns.append(decorator)
        
        # Strategy detection
        strategy = self._detect_strategy(cls, file_path)
        if strategy:
            patterns.append(strategy)
        
        # Builder detection
        builder = self._detect_builder(cls, file_path)
        if builder:
            patterns.append(builder)
        
        return patterns
    
    def _detect_singleton(self, cls: Class, file_path: str) -> Optional[PatternMatch]:
        """Detect Singleton pattern."""
        evidence = []
        confidence = 0.0
        
        method_names = [m.name for m in cls.methods]
        
        # Check for private instance variable
        for var in cls.class_variables:
            if var.name in ('_instance', '__instance', '_singleton'):
                evidence.append(f"Private instance variable: {var.name}")
                confidence += 0.3
        
        # Check for getInstance or similar method
        for m in cls.methods:
            if m.name in ('get_instance', 'getInstance', 'instance'):
                if m.is_classmethod or m.is_static:
                    evidence.append(f"Factory method: {m.name}")
                    confidence += 0.3
        
        # Check for __new__ override
        if '__new__' in method_names:
            evidence.append("Overrides __new__")
            confidence += 0.2
        
        # Check for private __init__
        for m in cls.methods:
            if m.name == '__init__':
                # Check if it raises or has special logic
                if 'raise' in str(m.calls):
                    evidence.append("Protected __init__")
                    confidence += 0.2
        
        if confidence >= 0.5:
            return PatternMatch(
                pattern_type=PatternType.SINGLETON,
                class_name=cls.name,
                file_path=file_path,
                line_number=cls.location.start_line,
                confidence=min(confidence, 1.0),
                evidence=evidence,
            )
        return None
    
    def _detect_factory(self, cls: Class, file_path: str) -> Optional[PatternMatch]:
        """Detect Factory pattern."""
        evidence = []
        confidence = 0.0
        
        # Check class name
        if 'Factory' in cls.name:
            evidence.append("Class name contains 'Factory'")
            confidence += 0.4
        
        # Check for create/make methods
        for m in cls.methods:
            if m.name.startswith(('create', 'make', 'build', 'get_')):
                # Check if method returns objects
                if m.return_type and m.return_type not in ('None', 'bool', 'int', 'str'):
                    evidence.append(f"Factory method: {m.name} -> {m.return_type}")
                    confidence += 0.3
        
        # Check for abstract methods suggesting abstract factory
        abstract_methods = [m for m in cls.methods if m.is_abstract]
        if abstract_methods:
            evidence.append(f"Abstract methods: {len(abstract_methods)}")
            confidence += 0.2
        
        if confidence >= 0.5:
            pattern_type = PatternType.ABSTRACT_FACTORY if cls.is_abstract else PatternType.FACTORY
            return PatternMatch(
                pattern_type=pattern_type,
                class_name=cls.name,
                file_path=file_path,
                line_number=cls.location.start_line,
                confidence=min(confidence, 1.0),
                evidence=evidence,
            )
        return None
    
    def _detect_observer(self, cls: Class, file_path: str) -> Optional[PatternMatch]:
        """Detect Observer pattern."""
        evidence = []
        confidence = 0.0
        
        method_names = [m.name for m in cls.methods]
        
        # Check for observer-related methods
        observer_methods = {
            'subscribe', 'unsubscribe', 'add_observer', 'remove_observer',
            'attach', 'detach', 'notify', 'notify_observers', 'register',
            'add_listener', 'remove_listener',
        }
        
        found_methods = set(method_names) & observer_methods
        if found_methods:
            evidence.append(f"Observer methods: {', '.join(found_methods)}")
            confidence += 0.4 * len(found_methods) / 3  # Cap at 3 methods
        
        # Check for observer list variable
        for var in cls.class_variables + cls.instance_variables:
            if var.name in ('observers', 'listeners', 'subscribers', '_observers'):
                evidence.append(f"Observer collection: {var.name}")
                confidence += 0.3
        
        # Check class name
        if any(x in cls.name for x in ('Observable', 'Subject', 'Publisher', 'EventEmitter')):
            evidence.append(f"Observable class name: {cls.name}")
            confidence += 0.2
        
        if confidence >= 0.4:
            return PatternMatch(
                pattern_type=PatternType.OBSERVER,
                class_name=cls.name,
                file_path=file_path,
                line_number=cls.location.start_line,
                confidence=min(confidence, 1.0),
                evidence=evidence,
            )
        return None
    
    def _detect_decorator(self, cls: Class, file_path: str) -> Optional[PatternMatch]:
        """Detect Decorator pattern (structural, not Python decorator)."""
        evidence = []
        confidence = 0.0
        
        # Check for wrapped component
        for var in cls.instance_variables:
            if var.name in ('_wrapped', '_component', '_decorated', 'component'):
                evidence.append(f"Wrapped component: {var.name}")
                confidence += 0.3
        
        # Check if class name suggests decorator
        if 'Decorator' in cls.name or 'Wrapper' in cls.name:
            evidence.append(f"Decorator class name: {cls.name}")
            confidence += 0.3
        
        # Check for delegation pattern in methods
        for m in cls.methods:
            # Look for calls to self._wrapped.method()
            for call in m.calls:
                if any(x in call for x in ('_wrapped.', '_component.', '_decorated.')):
                    evidence.append(f"Delegation in {m.name}")
                    confidence += 0.2
                    break
        
        if confidence >= 0.5:
            return PatternMatch(
                pattern_type=PatternType.DECORATOR,
                class_name=cls.name,
                file_path=file_path,
                line_number=cls.location.start_line,
                confidence=min(confidence, 1.0),
                evidence=evidence,
            )
        return None
    
    def _detect_strategy(self, cls: Class, file_path: str) -> Optional[PatternMatch]:
        """Detect Strategy pattern."""
        evidence = []
        confidence = 0.0
        
        # Check if abstract with single/few abstract methods
        if cls.is_abstract:
            abstract_methods = [m for m in cls.methods if m.is_abstract]
            if 1 <= len(abstract_methods) <= 3:
                evidence.append(f"Abstract strategy methods: {[m.name for m in abstract_methods]}")
                confidence += 0.4
        
        # Check class name
        if 'Strategy' in cls.name or 'Policy' in cls.name:
            evidence.append(f"Strategy class name: {cls.name}")
            confidence += 0.3
        
        # Check for execute/run/apply methods
        for m in cls.methods:
            if m.name in ('execute', 'run', 'apply', 'do', 'process'):
                evidence.append(f"Strategy method: {m.name}")
                confidence += 0.2
        
        if confidence >= 0.5:
            return PatternMatch(
                pattern_type=PatternType.STRATEGY,
                class_name=cls.name,
                file_path=file_path,
                line_number=cls.location.start_line,
                confidence=min(confidence, 1.0),
                evidence=evidence,
            )
        return None
    
    def _detect_builder(self, cls: Class, file_path: str) -> Optional[PatternMatch]:
        """Detect Builder pattern."""
        evidence = []
        confidence = 0.0
        
        # Check class name
        if 'Builder' in cls.name:
            evidence.append(f"Builder class name: {cls.name}")
            confidence += 0.4
        
        # Check for build method
        for m in cls.methods:
            if m.name in ('build', 'create', 'get_result', 'get_product'):
                evidence.append(f"Build method: {m.name}")
                confidence += 0.3
        
        # Check for fluent interface (methods returning self)
        fluent_methods = 0
        for m in cls.methods:
            if m.return_type and ('Self' in m.return_type or cls.name in m.return_type):
                fluent_methods += 1
        
        if fluent_methods >= 2:
            evidence.append(f"Fluent interface: {fluent_methods} methods return self")
            confidence += 0.3
        
        # Check for setter-like methods with set_ prefix
        set_methods = [m.name for m in cls.methods if m.name.startswith(('set_', 'with_'))]
        if len(set_methods) >= 2:
            evidence.append(f"Builder setters: {set_methods[:3]}")
            confidence += 0.2
        
        if confidence >= 0.5:
            return PatternMatch(
                pattern_type=PatternType.BUILDER,
                class_name=cls.name,
                file_path=file_path,
                line_number=cls.location.start_line,
                confidence=min(confidence, 1.0),
                evidence=evidence,
            )
        return None


def detect_design_patterns(modules: list[Module]) -> list[PatternMatch]:
    """
    Detect design patterns in modules.
    
    Args:
        modules: List of parsed modules
        
    Returns:
        List of detected patterns
    """
    detector = DesignPatternDetector()
    return detector.detect(modules)
