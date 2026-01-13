"""
Metrics data models.

Defines structured representations for code metrics:
- Complexity metrics (cyclomatic, cognitive)
- Lines of code metrics
- Maintainability index
- Halstead metrics
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum


class Rating(Enum):
    """Quality rating levels."""
    EXCELLENT = "A"
    GOOD = "B"
    MODERATE = "C"
    POOR = "D"
    CRITICAL = "F"


@dataclass
class ComplexityMetrics:
    """Complexity metrics for a code unit."""
    cyclomatic: int = 1
    cognitive: int = 0
    max_nesting_depth: int = 0
    
    @property
    def rating(self) -> Rating:
        """Get complexity rating."""
        if self.cyclomatic <= 5:
            return Rating.EXCELLENT
        elif self.cyclomatic <= 10:
            return Rating.GOOD
        elif self.cyclomatic <= 20:
            return Rating.MODERATE
        elif self.cyclomatic <= 30:
            return Rating.POOR
        return Rating.CRITICAL
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "cyclomatic": self.cyclomatic,
            "cognitive": self.cognitive,
            "max_nesting": self.max_nesting_depth,
            "rating": self.rating.value,
        }


@dataclass
class LOCMetrics:
    """Lines of code metrics."""
    total: int = 0
    source: int = 0  # Non-blank, non-comment
    comments: int = 0
    blank: int = 0
    docstrings: int = 0
    
    @property
    def comment_ratio(self) -> float:
        """Ratio of comments to source code."""
        if self.source == 0:
            return 0.0
        return self.comments / self.source
    
    @property
    def documentation_ratio(self) -> float:
        """Ratio of documentation (comments + docstrings) to source."""
        if self.source == 0:
            return 0.0
        return (self.comments + self.docstrings) / self.source
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "source": self.source,
            "comments": self.comments,
            "blank": self.blank,
            "docstrings": self.docstrings,
            "comment_ratio": round(self.comment_ratio, 3),
            "documentation_ratio": round(self.documentation_ratio, 3),
        }


@dataclass
class HalsteadMetrics:
    """Halstead complexity metrics."""
    # Operators and operands
    n1: int = 0  # Number of distinct operators
    n2: int = 0  # Number of distinct operands
    N1: int = 0  # Total number of operators
    N2: int = 0  # Total number of operands
    
    @property
    def vocabulary(self) -> int:
        """Program vocabulary (n = n1 + n2)."""
        return self.n1 + self.n2
    
    @property
    def length(self) -> int:
        """Program length (N = N1 + N2)."""
        return self.N1 + self.N2
    
    @property
    def calculated_length(self) -> float:
        """Calculated program length."""
        import math
        if self.n1 == 0 or self.n2 == 0:
            return 0.0
        return self.n1 * math.log2(self.n1) + self.n2 * math.log2(self.n2)
    
    @property
    def volume(self) -> float:
        """Program volume (V = N * log2(n))."""
        import math
        if self.vocabulary == 0:
            return 0.0
        return self.length * math.log2(self.vocabulary)
    
    @property
    def difficulty(self) -> float:
        """Program difficulty (D = (n1/2) * (N2/n2))."""
        if self.n2 == 0:
            return 0.0
        return (self.n1 / 2) * (self.N2 / self.n2)
    
    @property
    def effort(self) -> float:
        """Programming effort (E = D * V)."""
        return self.difficulty * self.volume
    
    @property
    def time_to_program(self) -> float:
        """Time to program in seconds (T = E / 18)."""
        return self.effort / 18
    
    @property
    def bugs(self) -> float:
        """Estimated number of bugs (B = V / 3000)."""
        return self.volume / 3000
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "vocabulary": self.vocabulary,
            "length": self.length,
            "volume": round(self.volume, 2),
            "difficulty": round(self.difficulty, 2),
            "effort": round(self.effort, 2),
            "time_to_program_seconds": round(self.time_to_program, 2),
            "estimated_bugs": round(self.bugs, 3),
        }


@dataclass
class MaintainabilityMetrics:
    """Maintainability index metrics."""
    halstead_volume: float = 0.0
    cyclomatic_complexity: int = 1
    loc: int = 0
    comment_ratio: float = 0.0
    
    @property
    def maintainability_index(self) -> float:
        """
        Calculate Maintainability Index.
        
        Original formula:
        MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)
        
        Microsoft formula (with comments):
        MI = MAX(0, (171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC) + 50 * sin(sqrt(2.4 * CM))) * 100 / 171)
        
        Returns:
            Maintainability index (0-100 scale)
        """
        import math
        
        if self.loc == 0 or self.halstead_volume == 0:
            return 100.0
        
        ln_volume = math.log(self.halstead_volume) if self.halstead_volume > 0 else 0
        ln_loc = math.log(self.loc) if self.loc > 0 else 0
        
        # Calculate base MI
        mi = 171 - 5.2 * ln_volume - 0.23 * self.cyclomatic_complexity - 16.2 * ln_loc
        
        # Add comment bonus
        if self.comment_ratio > 0:
            mi += 50 * math.sin(math.sqrt(2.4 * self.comment_ratio))
        
        # Normalize to 0-100 scale
        mi = max(0, mi * 100 / 171)
        
        return min(100, mi)
    
    @property
    def rating(self) -> Rating:
        """Get maintainability rating."""
        mi = self.maintainability_index
        if mi >= 85:
            return Rating.EXCELLENT
        elif mi >= 70:
            return Rating.GOOD
        elif mi >= 50:
            return Rating.MODERATE
        elif mi >= 25:
            return Rating.POOR
        return Rating.CRITICAL
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "maintainability_index": round(self.maintainability_index, 2),
            "rating": self.rating.value,
            "halstead_volume": round(self.halstead_volume, 2),
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "loc": self.loc,
            "comment_ratio": round(self.comment_ratio, 3),
        }


@dataclass
class QualityScore:
    """Overall quality score for a code unit."""
    complexity: ComplexityMetrics = field(default_factory=ComplexityMetrics)
    loc: LOCMetrics = field(default_factory=LOCMetrics)
    halstead: HalsteadMetrics = field(default_factory=HalsteadMetrics)
    maintainability: MaintainabilityMetrics = field(default_factory=MaintainabilityMetrics)
    
    # Issue counts
    code_smells: int = 0
    anti_patterns: int = 0
    security_issues: int = 0
    style_issues: int = 0
    
    @property
    def overall_rating(self) -> Rating:
        """Calculate overall quality rating."""
        # Weight different factors
        scores = {
            "complexity": self._rating_to_score(self.complexity.rating),
            "maintainability": self._rating_to_score(self.maintainability.rating),
        }
        
        # Deduct for issues
        issue_penalty = (
            self.code_smells * 2 +
            self.anti_patterns * 5 +
            self.security_issues * 10 +
            self.style_issues * 1
        )
        
        avg_score = sum(scores.values()) / len(scores) - issue_penalty
        
        if avg_score >= 85:
            return Rating.EXCELLENT
        elif avg_score >= 70:
            return Rating.GOOD
        elif avg_score >= 50:
            return Rating.MODERATE
        elif avg_score >= 25:
            return Rating.POOR
        return Rating.CRITICAL
    
    @staticmethod
    def _rating_to_score(rating: Rating) -> int:
        """Convert rating to numeric score."""
        return {
            Rating.EXCELLENT: 100,
            Rating.GOOD: 80,
            Rating.MODERATE: 60,
            Rating.POOR: 40,
            Rating.CRITICAL: 20,
        }[rating]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_rating": self.overall_rating.value,
            "complexity": self.complexity.to_dict(),
            "loc": self.loc.to_dict(),
            "halstead": self.halstead.to_dict(),
            "maintainability": self.maintainability.to_dict(),
            "issues": {
                "code_smells": self.code_smells,
                "anti_patterns": self.anti_patterns,
                "security_issues": self.security_issues,
                "style_issues": self.style_issues,
            },
        }


@dataclass
class FileMetrics:
    """Aggregated metrics for a file."""
    file_path: str
    quality: QualityScore = field(default_factory=QualityScore)
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    
    # Per-function metrics
    function_metrics: dict[str, QualityScore] = field(default_factory=dict)
    
    # Per-class metrics
    class_metrics: dict[str, QualityScore] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "quality": self.quality.to_dict(),
            "function_count": self.function_count,
            "class_count": self.class_count,
            "import_count": self.import_count,
            "functions": {k: v.to_dict() for k, v in self.function_metrics.items()},
            "classes": {k: v.to_dict() for k, v in self.class_metrics.items()},
        }


@dataclass
class ProjectMetrics:
    """Aggregated metrics for an entire project."""
    name: str
    root_path: str
    file_count: int = 0
    total_loc: int = 0
    quality: QualityScore = field(default_factory=QualityScore)
    file_metrics: dict[str, FileMetrics] = field(default_factory=dict)
    
    # Aggregated counts
    total_classes: int = 0
    total_functions: int = 0
    total_imports: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root_path": self.root_path,
            "file_count": self.file_count,
            "total_loc": self.total_loc,
            "quality": self.quality.to_dict(),
            "totals": {
                "classes": self.total_classes,
                "functions": self.total_functions,
                "imports": self.total_imports,
            },
            "files": {k: v.to_dict() for k, v in self.file_metrics.items()},
        }
