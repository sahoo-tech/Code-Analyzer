

from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum


class EntityType(Enum):
    """Types of code entities."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    ASYNC_FUNCTION = "async_function"
    ASYNC_METHOD = "async_method"
    PROPERTY = "property"
    STATIC_METHOD = "static_method"
    CLASS_METHOD = "class_method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    PARAMETER = "parameter"
    IMPORT = "import"
    DECORATOR = "decorator"


class Visibility(Enum):
    """Visibility/access level."""
    PUBLIC = "public"
    PROTECTED = "protected"  # Single underscore
    PRIVATE = "private"      # Double underscore


@dataclass
class CodeLocation:
    """Location of a code element in source."""
    file_path: str
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0
    
    @property
    def line_count(self) -> int:
        """Number of lines spanned."""
        return self.end_line - self.start_line + 1
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_col": self.start_col,
            "end_col": self.end_col,
        }


@dataclass
class Docstring:
    """Parsed docstring information."""
    raw: str
    summary: str = ""
    description: str = ""
    params: dict[str, str] = field(default_factory=dict)
    returns: Optional[str] = None
    raises: dict[str, str] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "description": self.description,
            "params": self.params,
            "returns": self.returns,
            "raises": self.raises,
            "examples": self.examples,
        }


@dataclass
class Decorator:
    """Decorator applied to a class or function."""
    name: str
    arguments: list[str] = field(default_factory=list)
    keyword_arguments: dict[str, str] = field(default_factory=dict)
    location: Optional[CodeLocation] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "keyword_arguments": self.keyword_arguments,
        }


@dataclass
class Parameter:
    """Function/method parameter."""
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None
    kind: str = "positional_or_keyword"  # positional_only, keyword_only, var_positional, var_keyword
    
    @property
    def is_optional(self) -> bool:
        return self.default_value is not None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type_annotation,
            "default": self.default_value,
            "kind": self.kind,
            "optional": self.is_optional,
        }


@dataclass
class Import:
    """Import statement."""
    module: str
    name: Optional[str] = None  # For 'from X import Y'
    alias: Optional[str] = None
    is_from_import: bool = False
    is_relative: bool = False
    level: int = 0  # Relative import level (number of dots)
    location: Optional[CodeLocation] = None
    
    @property
    def full_name(self) -> str:
        """Full imported name."""
        if self.name:
            return f"{self.module}.{self.name}"
        return self.module
    
    @property
    def used_name(self) -> str:
        """Name used in code (alias or original)."""
        if self.alias:
            return self.alias
        if self.name:
            return self.name
        return self.module
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "name": self.name,
            "alias": self.alias,
            "is_from_import": self.is_from_import,
            "is_relative": self.is_relative,
        }


@dataclass
class Variable:
    """Variable or constant."""
    name: str
    type_annotation: Optional[str] = None
    value: Optional[str] = None
    is_constant: bool = False
    is_class_variable: bool = False
    is_instance_variable: bool = False
    visibility: Visibility = Visibility.PUBLIC
    location: Optional[CodeLocation] = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type_annotation,
            "is_constant": self.is_constant,
            "visibility": self.visibility.value,
        }


@dataclass
class CodeEntity:
    """Base class for code entities."""
    name: str
    entity_type: EntityType
    location: CodeLocation
    docstring: Optional[Docstring] = None
    visibility: Visibility = Visibility.PUBLIC
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.entity_type.value,
            "location": self.location.to_dict(),
            "docstring": self.docstring.to_dict() if self.docstring else None,
            "visibility": self.visibility.value,
        }


@dataclass
class Function(CodeEntity):
    """Function definition."""
    parameters: list[Parameter] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: list[Decorator] = field(default_factory=list)
    is_async: bool = False
    is_generator: bool = False
    calls: list[str] = field(default_factory=list)  # Functions called
    variables: list[Variable] = field(default_factory=list)  # Local variables
    
    def __post_init__(self):
        if self.entity_type == EntityType.FUNCTION and self.is_async:
            self.entity_type = EntityType.ASYNC_FUNCTION
    
    @property
    def signature(self) -> str:
        """Get function signature."""
        params = ", ".join(
            f"{p.name}: {p.type_annotation}" if p.type_annotation else p.name
            for p in self.parameters
        )
        ret = f" -> {self.return_type}" if self.return_type else ""
        async_prefix = "async " if self.is_async else ""
        return f"{async_prefix}def {self.name}({params}){ret}"
    
    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "parameters": [p.to_dict() for p in self.parameters],
            "return_type": self.return_type,
            "decorators": [d.to_dict() for d in self.decorators],
            "is_async": self.is_async,
            "is_generator": self.is_generator,
            "signature": self.signature,
            "calls": self.calls,
        })
        return base


@dataclass
class Method(Function):
    """Method definition (function within a class)."""
    is_static: bool = False
    is_classmethod: bool = False
    is_property: bool = False
    is_abstract: bool = False
    overrides: Optional[str] = None  # Parent class method it overrides
    
    def __post_init__(self):
        if self.is_static:
            self.entity_type = EntityType.STATIC_METHOD
        elif self.is_classmethod:
            self.entity_type = EntityType.CLASS_METHOD
        elif self.is_property:
            self.entity_type = EntityType.PROPERTY
        elif self.is_async:
            self.entity_type = EntityType.ASYNC_METHOD
        else:
            self.entity_type = EntityType.METHOD
    
    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "is_static": self.is_static,
            "is_classmethod": self.is_classmethod,
            "is_property": self.is_property,
            "is_abstract": self.is_abstract,
            "overrides": self.overrides,
        })
        return base


@dataclass
class Class(CodeEntity):
    """Class definition."""
    bases: list[str] = field(default_factory=list)
    decorators: list[Decorator] = field(default_factory=list)
    methods: list[Method] = field(default_factory=list)
    class_variables: list[Variable] = field(default_factory=list)
    instance_variables: list[Variable] = field(default_factory=list)
    nested_classes: list["Class"] = field(default_factory=list)
    is_abstract: bool = False
    is_dataclass: bool = False
    metaclass: Optional[str] = None
    
    @property
    def all_methods(self) -> list[Method]:
        """All methods including nested."""
        methods = self.methods.copy()
        for nested in self.nested_classes:
            methods.extend(nested.all_methods)
        return methods
    
    @property
    def public_methods(self) -> list[Method]:
        """Public methods only."""
        return [m for m in self.methods if m.visibility == Visibility.PUBLIC]
    
    @property
    def properties(self) -> list[Method]:
        """Property methods."""
        return [m for m in self.methods if m.is_property]
    
    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "bases": self.bases,
            "decorators": [d.to_dict() for d in self.decorators],
            "methods": [m.to_dict() for m in self.methods],
            "class_variables": [v.to_dict() for v in self.class_variables],
            "instance_variables": [v.to_dict() for v in self.instance_variables],
            "nested_classes": [c.to_dict() for c in self.nested_classes],
            "is_abstract": self.is_abstract,
            "is_dataclass": self.is_dataclass,
            "method_count": len(self.methods),
            "variable_count": len(self.class_variables) + len(self.instance_variables),
        })
        return base


@dataclass
class Module(CodeEntity):
    """Module (file) definition."""
    file_path: str = ""
    package: Optional[str] = None
    imports: list[Import] = field(default_factory=list)
    classes: list[Class] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)
    variables: list[Variable] = field(default_factory=list)
    constants: list[Variable] = field(default_factory=list)
    
    @property
    def all_names(self) -> list[str]:
        """All top-level names defined in module."""
        names = []
        names.extend(c.name for c in self.classes)
        names.extend(f.name for f in self.functions)
        names.extend(v.name for v in self.variables)
        names.extend(c.name for c in self.constants)
        return names
    
    @property
    def exported_names(self) -> list[str]:
        """Names that would be exported (public only)."""
        return [name for name in self.all_names if not name.startswith("_")]
    
    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "file_path": self.file_path,
            "package": self.package,
            "imports": [i.to_dict() for i in self.imports],
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions],
            "variables": [v.to_dict() for v in self.variables],
            "constants": [c.to_dict() for c in self.constants],
            "summary": {
                "class_count": len(self.classes),
                "function_count": len(self.functions),
                "import_count": len(self.imports),
            },
        })
        return base
