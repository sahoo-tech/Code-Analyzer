"""
Maintainability Index calculator.

Calculates the Maintainability Index (MI) which is a composite metric
combining Halstead Volume, Cyclomatic Complexity, and Lines of Code.
"""

from typing import Optional

from analyzer.metrics.complexity import CyclomaticComplexityCalculator
from analyzer.metrics.loc import LOCCalculator
from analyzer.metrics.halstead import HalsteadCalculator
from analyzer.models.metrics import MaintainabilityMetrics, QualityScore
from analyzer.logging_config import get_logger

logger = get_logger("metrics.maintainability")


class MaintainabilityCalculator:
    """
    Maintainability Index calculator.
    
    Uses the Microsoft variant of the Maintainability Index formula:
    MI = MAX(0, (171 - 5.2*ln(V) - 0.23*G - 16.2*ln(LOC) + 50*sin(sqrt(2.4*CM))) * 100/171)
    
    Where:
        V = Halstead Volume
        G = Cyclomatic Complexity
        LOC = Source Lines of Code
        CM = Comment Ratio (percent of comments)
    """
    
    def __init__(self):
        self.halstead_calc = HalsteadCalculator()
        self.complexity_calc = CyclomaticComplexityCalculator()
        self.loc_calc = LOCCalculator()
    
    def calculate(self, code: str) -> MaintainabilityMetrics:
        """
        Calculate maintainability metrics for code.
        
        Args:
            code: Source code string
            
        Returns:
            MaintainabilityMetrics with calculated index
        """
        # Calculate component metrics
        halstead = self.halstead_calc.calculate(code)
        cyclomatic = self.complexity_calc.calculate(code)
        loc = self.loc_calc.calculate(code)
        
        # Calculate comment ratio
        comment_ratio = 0.0
        if loc.source > 0:
            comment_ratio = (loc.comments + loc.docstrings) / loc.total
        
        return MaintainabilityMetrics(
            halstead_volume=halstead.volume,
            cyclomatic_complexity=cyclomatic,
            loc=loc.source,
            comment_ratio=comment_ratio,
        )
    
    def calculate_quality_score(self, code: str) -> QualityScore:
        """
        Calculate comprehensive quality score for code.
        
        Args:
            code: Source code string
            
        Returns:
            QualityScore with all metrics
        """
        from analyzer.metrics.complexity import calculate_complexity
        from analyzer.metrics.loc import calculate_loc
        from analyzer.metrics.halstead import calculate_halstead
        
        complexity = calculate_complexity(code)
        loc = calculate_loc(code)
        halstead = calculate_halstead(code)
        maintainability = self.calculate(code)
        
        return QualityScore(
            complexity=complexity,
            loc=loc,
            halstead=halstead,
            maintainability=maintainability,
        )


def calculate_maintainability(code: str) -> MaintainabilityMetrics:
    """
    Calculate maintainability metrics for code.
    
    Args:
        code: Source code string
        
    Returns:
        MaintainabilityMetrics with calculated index
    """
    return MaintainabilityCalculator().calculate(code)


def calculate_quality(code: str) -> QualityScore:
    """
    Calculate comprehensive quality score for code.
    
    Args:
        code: Source code string
        
    Returns:
        QualityScore with all metrics
    """
    return MaintainabilityCalculator().calculate_quality_score(code)
