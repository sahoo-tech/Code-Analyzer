

from typing import Optional

from analyzer.metrics.complexity import CyclomaticComplexityCalculator
from analyzer.metrics.loc import LOCCalculator
from analyzer.metrics.halstead import HalsteadCalculator
from analyzer.models.metrics import MaintainabilityMetrics, QualityScore
from analyzer.logging_config import get_logger

logger = get_logger("metrics.maintainability")


class MaintainabilityCalculator:

    
    def __init__(self):
        self.halstead_calc = HalsteadCalculator()
        self.complexity_calc = CyclomaticComplexityCalculator()
        self.loc_calc = LOCCalculator()
    
    def calculate(self, code: str) -> MaintainabilityMetrics:
 
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

    return MaintainabilityCalculator().calculate(code)


def calculate_quality(code: str) -> QualityScore:

    return MaintainabilityCalculator().calculate_quality_score(code)
