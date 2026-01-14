
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from analyzer.models.code_entities import Module, Class, Function, Method
from analyzer.rag.config import ChunkingConfig
from analyzer.logging_config import get_logger

logger = get_logger("rag.chunker")


@dataclass
class CodeChunk:
    """A chunk of code with metadata for embedding and retrieval."""
    
    content: str
    chunk_id: str
    
    # Source information
    file_path: str
    module_name: str
    
    # Entity information
    entity_type: str  # "module", "class", "function", "method"
    entity_name: str
    
    # Location
    start_line: int
    end_line: int
    
    # Additional metadata
    metadata: dict = field(default_factory=dict)
    
    # Parent entity (for methods)
    parent_name: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "file_path": self.file_path,
            "module_name": self.module_name,
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "parent_name": self.parent_name,
            **self.metadata,
        }
    
    @property
    def full_name(self) -> str:
        """Get fully qualified name."""
        if self.parent_name:
            return f"{self.module_name}.{self.parent_name}.{self.entity_name}"
        return f"{self.module_name}.{self.entity_name}"
    
    def __repr__(self) -> str:
        return f"CodeChunk({self.entity_type}:{self.full_name}, lines {self.start_line}-{self.end_line})"


class CodeChunker:
 
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
    
    def chunk_modules(self, modules: list[Module]) -> list[CodeChunk]:
      
        chunks = []
        for module in modules:
            chunks.extend(self.chunk_module(module))
        
        logger.info(f"Created {len(chunks)} chunks from {len(modules)} modules")
        return chunks
    
    def chunk_module(self, module: Module) -> list[CodeChunk]:
  
        chunks = []
        
        # Create module overview chunk
        module_chunk = self._create_module_chunk(module)
        if module_chunk:
            chunks.append(module_chunk)
        
        # Chunk each class
        for cls in module.classes:
            chunks.extend(self._chunk_class(cls, module))
        
        # Chunk standalone functions
        for func in module.functions:
            func_chunk = self._create_function_chunk(func, module)
            if func_chunk:
                chunks.append(func_chunk)
        
        return chunks
    
    def _create_module_chunk(self, module: Module) -> Optional[CodeChunk]:
        """Create a chunk for module-level information."""
        parts = []
        
        # Module docstring
        if module.docstring and self.config.include_docstrings:
            parts.append(f'"""Module: {module.name}"""')
            parts.append(f"Description: {module.docstring.summary}")
            if module.docstring.description:
                parts.append(module.docstring.description[:500])
        else:
            parts.append(f"Module: {module.name}")
        
        # File path
        parts.append(f"File: {module.file_path}")
        
        # Import summary
        if self.config.include_metadata and module.imports:
            stdlib_imports = []
            third_party = []
            local_imports = []
            
            for imp in module.imports:
                if imp.is_relative:
                    local_imports.append(imp.module)
                else:
                    # Simplified categorization
                    stdlib_imports.append(imp.module)
            
            if stdlib_imports:
                parts.append(f"Imports: {', '.join(list(set(stdlib_imports))[:10])}")
            if local_imports:
                parts.append(f"Local imports: {', '.join(list(set(local_imports))[:5])}")
        
        # Summary of contents
        parts.append(f"Contains: {len(module.classes)} classes, {len(module.functions)} functions")
        
        # List main entities
        if module.classes:
            class_names = [cls.name for cls in module.classes[:5]]
            parts.append(f"Classes: {', '.join(class_names)}")
        
        if module.functions:
            func_names = [f.name for f in module.functions[:5]]
            parts.append(f"Functions: {', '.join(func_names)}")
        
        content = "\n".join(parts)
        
        if len(content) < 50:  # Skip nearly empty modules
            return None
        
        return CodeChunk(
            content=content,
            chunk_id=self._generate_chunk_id(module.file_path, "module", module.name),
            file_path=module.file_path,
            module_name=module.name,
            entity_type="module",
            entity_name=module.name,
            start_line=1,
            end_line=module.classes[0].location.start_line if module.classes else 1,
            metadata={
                "class_count": len(module.classes),
                "function_count": len(module.functions),
                "import_count": len(module.imports),
            },
        )
    
    def _chunk_class(self, cls: Class, module: Module) -> list[CodeChunk]:
        """Chunk a class, potentially splitting into multiple chunks."""
        chunks = []
        
        # Create class overview chunk
        class_chunk = self._create_class_overview_chunk(cls, module)
        if class_chunk:
            chunks.append(class_chunk)
        
        # Chunk each method separately if class is large
        if self.config.entity_based_chunking and len(cls.methods) > 3:
            for method in cls.methods:
                method_chunk = self._create_method_chunk(method, cls, module)
                if method_chunk:
                    chunks.append(method_chunk)
        
        return chunks
    
    def _create_class_overview_chunk(self, cls: Class, module: Module) -> Optional[CodeChunk]:
        """Create a chunk with class overview information."""
        parts = []
        
        # Class declaration
        bases = f"({', '.join(cls.bases)})" if cls.bases else ""
        parts.append(f"class {cls.name}{bases}:")
        
        # Docstring
        if cls.docstring and self.config.include_docstrings:
            parts.append(f'    """{cls.docstring.summary}"""')
            if cls.docstring.description:
                desc = cls.docstring.description[:300]
                parts.append(f"    {desc}")
        
        # Class properties
        properties = []
        if cls.is_abstract:
            properties.append("abstract")
        if cls.is_dataclass:
            properties.append("dataclass")
        if properties:
            parts.append(f"    # Properties: {', '.join(properties)}")
        
        # Class variables
        if cls.class_variables:
            parts.append(f"    # Class variables: {len(cls.class_variables)}")
            for var in cls.class_variables[:5]:
                type_hint = f": {var.type_annotation}" if var.type_annotation else ""
                parts.append(f"    {var.name}{type_hint}")
        
        # Methods summary
        if cls.methods:
            parts.append(f"    # Methods ({len(cls.methods)}):")
            for method in cls.methods:
                sig = self._get_method_signature(method)
                parts.append(f"    {sig}")
        
        content = "\n".join(parts)
        
        # Truncate if too long
        if len(content) > self.config.chunk_size:
            content = content[:self.config.chunk_size - 3] + "..."
        
        return CodeChunk(
            content=content,
            chunk_id=self._generate_chunk_id(module.file_path, "class", cls.name),
            file_path=module.file_path,
            module_name=module.name,
            entity_type="class",
            entity_name=cls.name,
            start_line=cls.location.start_line,
            end_line=cls.location.end_line,
            metadata={
                "bases": cls.bases,
                "is_abstract": cls.is_abstract,
                "is_dataclass": cls.is_dataclass,
                "method_count": len(cls.methods),
            },
        )
    
    def _create_method_chunk(
        self, method: Method, cls: Class, module: Module
    ) -> Optional[CodeChunk]:
        """Create a chunk for a single method."""
        parts = []
        
        # Context
        parts.append(f"# Method in class {cls.name}")
        
        # Decorators
        if method.decorators:
            for dec in method.decorators:
                parts.append(f"@{dec.name}")
        
        # Method signature
        sig = self._get_method_signature(method, include_types=True)
        parts.append(sig)
        
        # Docstring
        if method.docstring and self.config.include_docstrings:
            parts.append(f'    """{method.docstring.summary}')
            
            if method.docstring.params:
                parts.append("    ")
                parts.append("    Args:")
                for param, desc in list(method.docstring.params.items())[:5]:
                    parts.append(f"        {param}: {desc[:80]}")
            
            if method.docstring.returns:
                parts.append(f"    Returns: {method.docstring.returns[:100]}")
            
            parts.append('    """')
        
        # Method properties
        props = []
        if method.is_async:
            props.append("async")
        if method.is_static:
            props.append("static")
        if method.is_classmethod:
            props.append("classmethod")
        if method.is_abstract:
            props.append("abstract")
        if props:
            parts.append(f"    # {', '.join(props)}")
        
        content = "\n".join(parts)
        
        return CodeChunk(
            content=content,
            chunk_id=self._generate_chunk_id(
                module.file_path, "method", f"{cls.name}.{method.name}"
            ),
            file_path=module.file_path,
            module_name=module.name,
            entity_type="method",
            entity_name=method.name,
            parent_name=cls.name,
            start_line=method.location.start_line,
            end_line=method.location.end_line,
            metadata={
                "is_async": method.is_async,
                "is_static": method.is_static,
                "is_abstract": method.is_abstract,
                "param_count": len(method.parameters),
            },
        )
    
    def _create_function_chunk(
        self, func: Function, module: Module
    ) -> Optional[CodeChunk]:
        """Create a chunk for a standalone function."""
        parts = []
        
        # Decorators
        if func.decorators:
            for dec in func.decorators:
                parts.append(f"@{dec.name}")
        
        # Function signature with types
        params = []
        for p in func.parameters:
            if p.type_annotation:
                params.append(f"{p.name}: {p.type_annotation}")
            elif p.default_value:
                params.append(f"{p.name}={p.default_value}")
            else:
                params.append(p.name)
        
        async_prefix = "async " if func.is_async else ""
        ret_type = f" -> {func.return_type}" if func.return_type else ""
        sig = f"{async_prefix}def {func.name}({', '.join(params)}){ret_type}:"
        parts.append(sig)
        
        # Docstring
        if func.docstring and self.config.include_docstrings:
            parts.append(f'    """{func.docstring.summary}')
            
            if func.docstring.description:
                parts.append(f"    {func.docstring.description[:200]}")
            
            if func.docstring.params:
                parts.append("    Args:")
                for param, desc in list(func.docstring.params.items())[:5]:
                    parts.append(f"        {param}: {desc[:80]}")
            
            if func.docstring.returns:
                parts.append(f"    Returns: {func.docstring.returns[:100]}")
            
            parts.append('    """')
        
        content = "\n".join(parts)
        
        return CodeChunk(
            content=content,
            chunk_id=self._generate_chunk_id(module.file_path, "function", func.name),
            file_path=module.file_path,
            module_name=module.name,
            entity_type="function",
            entity_name=func.name,
            start_line=func.location.start_line,
            end_line=func.location.end_line,
            metadata={
                "is_async": func.is_async,
                "is_generator": func.is_generator,
                "param_count": len(func.parameters),
                "return_type": func.return_type,
            },
        )
    
    def _get_method_signature(
        self, method: Method, include_types: bool = False
    ) -> str:
        """Generate method signature string."""
        params = []
        for p in method.parameters:
            if include_types and p.type_annotation:
                params.append(f"{p.name}: {p.type_annotation}")
            else:
                params.append(p.name)
        
        async_prefix = "async " if method.is_async else ""
        return f"{async_prefix}def {method.name}({', '.join(params)})"
    
    def _generate_chunk_id(
        self, file_path: str, entity_type: str, entity_name: str
    ) -> str:
        """Generate unique chunk ID."""
        content = f"{file_path}:{entity_type}:{entity_name}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
