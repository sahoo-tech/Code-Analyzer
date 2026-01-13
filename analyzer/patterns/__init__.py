"""Patterns module initialization."""

from analyzer.patterns.design_patterns import DesignPatternDetector, detect_design_patterns
from analyzer.patterns.anti_patterns import AntiPatternDetector, detect_anti_patterns
from analyzer.patterns.code_smells import CodeSmellDetector, detect_code_smells
from analyzer.patterns.dead_code import DeadCodeDetector, detect_dead_code
from analyzer.patterns.duplicates import DuplicateDetector, detect_duplicates

__all__ = [
    "DesignPatternDetector",
    "detect_design_patterns",
    "AntiPatternDetector",
    "detect_anti_patterns",
    "CodeSmellDetector",
    "detect_code_smells",
    "DeadCodeDetector",
    "detect_dead_code",
    "DuplicateDetector",
    "detect_duplicates",
]
