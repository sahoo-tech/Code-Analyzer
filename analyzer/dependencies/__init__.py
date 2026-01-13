"""Dependencies module initialization."""

from analyzer.dependencies.import_analyzer import ImportAnalyzer, analyze_imports
from analyzer.dependencies.call_graph import CallGraphBuilder, build_call_graph
from analyzer.dependencies.module_graph import ModuleGraphBuilder, build_module_graph

__all__ = [
    "ImportAnalyzer",
    "analyze_imports",
    "CallGraphBuilder",
    "build_call_graph",
    "ModuleGraphBuilder",
    "build_module_graph",
]
