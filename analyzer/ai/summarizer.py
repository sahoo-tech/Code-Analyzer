

from dataclasses import dataclass
from typing import Optional

from analyzer.models.code_entities import Module, Class, Function
from analyzer.logging_config import get_logger

logger = get_logger("ai.summarizer")


@dataclass
class SummaryConfig:
    """Configuration for code summarization."""
    max_length: int = 2000
    include_docstrings: bool = True
    include_signatures: bool = True
    include_imports: bool = True
    include_metrics: bool = True
    detail_level: str = "summary"  # minimal, summary, detailed


class CodeSummarizer:
    """Creates concise code summaries for AI consumption."""
    
    def __init__(self, config: Optional[SummaryConfig] = None):
        self.config = config or SummaryConfig()
    
    def summarize_module(self, module: Module) -> str:
  
        parts = []
        
        # Header
        parts.append(f"Module: {module.name}")
        parts.append(f"Path: {module.file_path}")
        
        # Docstring
        if module.docstring and self.config.include_docstrings:
            parts.append(f"Purpose: {module.docstring.summary}")
        
        # Statistics
        stats = [
            f"{len(module.classes)} classes",
            f"{len(module.functions)} functions",
            f"{len(module.imports)} imports",
        ]
        parts.append(f"Contains: {', '.join(stats)}")
        
        # Imports
        if self.config.include_imports and module.imports:
            main_imports = [imp.module for imp in module.imports[:5]]
            if len(module.imports) > 5:
                main_imports.append(f"...and {len(module.imports) - 5} more")
            parts.append(f"Imports: {', '.join(main_imports)}")
        
        # Classes
        if module.classes:
            parts.append("\nClasses:")
            for cls in module.classes:
                parts.append(self._summarize_class(cls))
        
        # Functions
        if module.functions:
            parts.append("\nFunctions:")
            for func in module.functions[:10]:
                parts.append(self._summarize_function(func))
            if len(module.functions) > 10:
                parts.append(f"  ...and {len(module.functions) - 10} more functions")
        
        # Truncate if needed
        summary = "\n".join(parts)
        if len(summary) > self.config.max_length:
            summary = summary[:self.config.max_length - 3] + "..."
        
        return summary
    
    def summarize_class(self, cls: Class) -> str:
        """Create a detailed summary of a class."""
        parts = []
        
        # Header
        bases = f"({', '.join(cls.bases)})" if cls.bases else ""
        parts.append(f"class {cls.name}{bases}:")
        
        # Docstring
        if cls.docstring and self.config.include_docstrings:
            parts.append(f"  \"\"\"{cls.docstring.summary}\"\"\"")
        
        # Properties
        if cls.is_abstract:
            parts.append("  # Abstract class")
        if cls.is_dataclass:
            parts.append("  # Dataclass")
        
        # Class variables
        if cls.class_variables:
            parts.append(f"  # {len(cls.class_variables)} class variables")
        
        # Methods
        if cls.methods:
            parts.append(f"  # {len(cls.methods)} methods:")
            for method in cls.methods[:8]:
                signature = self._get_signature(method)
                parts.append(f"    {signature}")
            if len(cls.methods) > 8:
                parts.append(f"    # ...and {len(cls.methods) - 8} more")
        
        return "\n".join(parts)
    
    def summarize_function(self, func: Function) -> str:
        """Create a detailed summary of a function."""
        parts = []
        
        # Signature
        signature = self._get_full_signature(func)
        parts.append(signature)
        
        # Docstring
        if func.docstring and self.config.include_docstrings:
            parts.append(f'    """{func.docstring.summary}')
            
            if func.docstring.params:
                parts.append("    ")
                parts.append("    Args:")
                for param, desc in list(func.docstring.params.items())[:5]:
                    parts.append(f"        {param}: {desc[:50]}")
            
            if func.docstring.returns:
                parts.append(f"    Returns: {func.docstring.returns[:50]}")
            
            parts.append('    """')
        
        return "\n".join(parts)
    
    def summarize_project(self, modules: list[Module]) -> str:
   
        parts = []
        
        # Overview statistics
        total_classes = sum(len(m.classes) for m in modules)
        total_functions = sum(len(m.functions) for m in modules)
        total_imports = sum(len(m.imports) for m in modules)
        
        parts.append("PROJECT OVERVIEW")
        parts.append("=" * 40)
        parts.append(f"Files: {len(modules)}")
        parts.append(f"Classes: {total_classes}")
        parts.append(f"Functions: {total_functions}")
        parts.append(f"Import statements: {total_imports}")
        parts.append("")
        
        # Module list
        parts.append("MODULES:")
        for module in modules[:20]:
            summary = f"  {module.name}: "
            items = []
            if module.classes:
                items.append(f"{len(module.classes)} classes")
            if module.functions:
                items.append(f"{len(module.functions)} functions")
            summary += ", ".join(items) if items else "empty"
            parts.append(summary)
        
        if len(modules) > 20:
            parts.append(f"  ...and {len(modules) - 20} more modules")
        
        parts.append("")
        
        # Key classes
        all_classes = [(m.name, cls) for m in modules for cls in m.classes]
        if all_classes:
            parts.append("KEY CLASSES:")
            for module_name, cls in all_classes[:10]:
                methods = len(cls.methods)
                parts.append(f"  {module_name}.{cls.name} ({methods} methods)")
        
        return "\n".join(parts)
    
    def _summarize_class(self, cls: Class) -> str:
        """Brief class summary for module context."""
        methods = len(cls.methods)
        bases = f" extends {', '.join(cls.bases)}" if cls.bases else ""
        doc = f" - {cls.docstring.summary[:40]}..." if cls.docstring else ""
        return f"  {cls.name}{bases}: {methods} methods{doc}"
    
    def _summarize_function(self, func: Function) -> str:
        """Brief function summary for module context."""
        params = len(func.parameters)
        async_prefix = "async " if func.is_async else ""
        doc = f" - {func.docstring.summary[:30]}..." if func.docstring else ""
        return f"  {async_prefix}{func.name}({params} params){doc}"
    
    def _get_signature(self, func: Function) -> str:
        """Get function signature."""
        params = ", ".join(p.name for p in func.parameters)
        async_prefix = "async " if func.is_async else ""
        return f"{async_prefix}def {func.name}({params})"
    
    def _get_full_signature(self, func: Function) -> str:
        """Get full function signature with types."""
        parts = []
        for p in func.parameters:
            if p.type_annotation:
                parts.append(f"{p.name}: {p.type_annotation}")
            else:
                parts.append(p.name)
        
        params = ", ".join(parts)
        async_prefix = "async " if func.is_async else ""
        ret = f" -> {func.return_type}" if func.return_type else ""
        
        return f"{async_prefix}def {func.name}({params}){ret}:"


def summarize_module(
    module: Module,
    config: Optional[SummaryConfig] = None
) -> str:

    summarizer = CodeSummarizer(config)
    return summarizer.summarize_module(module)


def summarize_project(
    modules: list[Module],
    config: Optional[SummaryConfig] = None
) -> str:

    summarizer = CodeSummarizer(config)
    return summarizer.summarize_project(modules)
