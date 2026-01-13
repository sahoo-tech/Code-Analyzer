"""
Python AST Parser.

Comprehensive Python source code parser using the AST module.
Extracts all code entities with full metadata.
"""

import ast
from pathlib import Path
from typing import Optional, Union, Any

from analyzer.parsers.base import BaseParser
from analyzer.models.code_entities import (
    Module, Class, Function, Method, Variable, Parameter,
    Import, Decorator, Docstring, CodeLocation, 
    EntityType, Visibility
)
from analyzer.exceptions import ParsingError, SyntaxParsingError, EncodingError
from analyzer.utils import read_file
from analyzer.logging_config import get_logger

logger = get_logger("parsers.python")


class PythonParser(BaseParser):
    """Parser for Python source code using AST."""
    
    @property
    def supported_extensions(self) -> list[str]:
        return [".py", ".pyw"]
    
    @property
    def language(self) -> str:
        return "python"
    
    def parse_file(self, path: Union[str, Path]) -> Module:
        """Parse a Python source file."""
        path = Path(path)
        
        try:
            code = read_file(path)
            module = self.parse_code(code, str(path))
            module.file_path = str(path)
            return module
        except EncodingError:
            raise
        except Exception as e:
            raise ParsingError(f"Failed to parse {path}: {e}")
    
    def parse_code(self, code: str, filename: str = "<string>") -> Module:
        """Parse Python source code string."""
        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError as e:
            raise SyntaxParsingError(
                f"Syntax error: {e.msg}",
                file_path=filename,
                line=e.lineno,
                column=e.offset
            )
        
        visitor = PythonASTVisitor(code, filename)
        visitor.visit(tree)
        
        return visitor.module


class PythonASTVisitor(ast.NodeVisitor):
    """Enhanced AST visitor that extracts all code entities."""
    
    def __init__(self, source: str, filename: str):
        self.source = source
        self.source_lines = source.splitlines()
        self.filename = filename
        
        # Create module
        self.module = Module(
            name=Path(filename).stem,
            entity_type=EntityType.MODULE,
            location=CodeLocation(
                file_path=filename,
                start_line=1,
                end_line=len(self.source_lines),
            ),
            file_path=filename,
        )
        
        # Track current context
        self._current_class: Optional[Class] = None
        self._scope_stack: list[str] = []
    
    def visit_Module(self, node: ast.Module) -> None:
        """Visit module node."""
        # Extract module docstring
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            self.module.docstring = self._parse_docstring(node.body[0].value.value)
        
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        for alias in node.names:
            import_obj = Import(
                module=alias.name,
                alias=alias.asname,
                is_from_import=False,
                location=self._get_location(node),
            )
            self.module.imports.append(import_obj)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statement."""
        module_name = node.module or ""
        
        for alias in node.names:
            import_obj = Import(
                module=module_name,
                name=alias.name,
                alias=alias.asname,
                is_from_import=True,
                is_relative=node.level > 0,
                level=node.level,
                location=self._get_location(node),
            )
            self.module.imports.append(import_obj)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        class_obj = Class(
            name=node.name,
            entity_type=EntityType.CLASS,
            location=self._get_location(node),
            visibility=self._get_visibility(node.name),
            bases=[self._get_name(base) for base in node.bases],
            decorators=[self._parse_decorator(d) for d in node.decorator_list],
        )
        
        # Extract docstring
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            class_obj.docstring = self._parse_docstring(node.body[0].value.value)
        
        # Check for special decorators
        for decorator in class_obj.decorators:
            if decorator.name == "dataclass":
                class_obj.is_dataclass = True
            elif decorator.name in ("abstractmethod", "ABC"):
                class_obj.is_abstract = True
        
        # Check bases for ABC
        if "ABC" in class_obj.bases or "abc.ABC" in class_obj.bases:
            class_obj.is_abstract = True
        
        # Visit class body
        old_class = self._current_class
        self._current_class = class_obj
        self._scope_stack.append(node.name)
        
        for child in node.body:
            if isinstance(child, ast.FunctionDef) or isinstance(child, ast.AsyncFunctionDef):
                method = self._parse_method(child)
                class_obj.methods.append(method)
            elif isinstance(child, ast.ClassDef):
                # Nested class
                self.visit_ClassDef(child)
                if self.module.classes and self.module.classes[-1].name == child.name:
                    nested = self.module.classes.pop()
                    class_obj.nested_classes.append(nested)
            elif isinstance(child, ast.Assign):
                for var in self._parse_assignment(child, is_class_var=True):
                    class_obj.class_variables.append(var)
            elif isinstance(child, ast.AnnAssign):
                var = self._parse_annotated_assignment(child, is_class_var=True)
                if var:
                    class_obj.class_variables.append(var)
        
        self._scope_stack.pop()
        self._current_class = old_class
        
        self.module.classes.append(class_obj)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        if self._current_class is None:
            func = self._parse_function(node)
            self.module.functions.append(func)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        if self._current_class is None:
            func = self._parse_function(node, is_async=True)
            self.module.functions.append(func)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment (module-level variables)."""
        if self._current_class is None and len(self._scope_stack) == 0:
            for var in self._parse_assignment(node):
                if var.is_constant:
                    self.module.constants.append(var)
                else:
                    self.module.variables.append(var)
    
    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Visit annotated assignment."""
        if self._current_class is None and len(self._scope_stack) == 0:
            var = self._parse_annotated_assignment(node)
            if var:
                if var.is_constant:
                    self.module.constants.append(var)
                else:
                    self.module.variables.append(var)
    
    def _parse_function(
        self, 
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        is_async: bool = False
    ) -> Function:
        """Parse a function definition."""
        is_async = is_async or isinstance(node, ast.AsyncFunctionDef)
        
        func = Function(
            name=node.name,
            entity_type=EntityType.ASYNC_FUNCTION if is_async else EntityType.FUNCTION,
            location=self._get_location(node),
            visibility=self._get_visibility(node.name),
            parameters=self._parse_parameters(node.args),
            return_type=self._get_annotation(node.returns),
            decorators=[self._parse_decorator(d) for d in node.decorator_list],
            is_async=is_async,
            is_generator=self._is_generator(node),
        )
        
        # Extract docstring
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            func.docstring = self._parse_docstring(node.body[0].value.value)
        
        # Extract function calls
        func.calls = self._extract_calls(node)
        
        # Extract local variables
        func.variables = self._extract_local_variables(node)
        
        return func
    
    def _parse_method(
        self, 
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> Method:
        """Parse a method definition."""
        is_async = isinstance(node, ast.AsyncFunctionDef)
        decorator_names = [self._get_name(d) for d in node.decorator_list]
        
        method = Method(
            name=node.name,
            entity_type=EntityType.METHOD,
            location=self._get_location(node),
            visibility=self._get_visibility(node.name),
            parameters=self._parse_parameters(node.args),
            return_type=self._get_annotation(node.returns),
            decorators=[self._parse_decorator(d) for d in node.decorator_list],
            is_async=is_async,
            is_generator=self._is_generator(node),
            is_static="staticmethod" in decorator_names,
            is_classmethod="classmethod" in decorator_names,
            is_property="property" in decorator_names,
            is_abstract="abstractmethod" in decorator_names,
        )
        
        # Extract docstring
        if (node.body and 
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            method.docstring = self._parse_docstring(node.body[0].value.value)
        
        # Extract function calls
        method.calls = self._extract_calls(node)
        
        return method
    
    def _parse_parameters(self, args: ast.arguments) -> list[Parameter]:
        """Parse function parameters."""
        params = []
        
        # Calculate defaults offset
        num_defaults = len(args.defaults)
        num_args = len(args.args)
        defaults_offset = num_args - num_defaults
        
        # Regular positional/keyword arguments
        for i, arg in enumerate(args.args):
            default = None
            if i >= defaults_offset:
                default = self._get_source(args.defaults[i - defaults_offset])
            
            params.append(Parameter(
                name=arg.arg,
                type_annotation=self._get_annotation(arg.annotation),
                default_value=default,
                kind="positional_or_keyword",
            ))
        
        # Positional-only arguments (Python 3.8+)
        if hasattr(args, 'posonlyargs'):
            for arg in args.posonlyargs:
                params.insert(0, Parameter(
                    name=arg.arg,
                    type_annotation=self._get_annotation(arg.annotation),
                    kind="positional_only",
                ))
        
        # *args
        if args.vararg:
            params.append(Parameter(
                name=args.vararg.arg,
                type_annotation=self._get_annotation(args.vararg.annotation),
                kind="var_positional",
            ))
        
        # Keyword-only arguments
        kw_defaults_offset = len(args.kw_defaults) - len(args.kwonlyargs)
        for i, arg in enumerate(args.kwonlyargs):
            default = None
            default_index = i + kw_defaults_offset
            if default_index >= 0 and args.kw_defaults[default_index] is not None:
                default = self._get_source(args.kw_defaults[default_index])
            
            params.append(Parameter(
                name=arg.arg,
                type_annotation=self._get_annotation(arg.annotation),
                default_value=default,
                kind="keyword_only",
            ))
        
        # **kwargs
        if args.kwarg:
            params.append(Parameter(
                name=args.kwarg.arg,
                type_annotation=self._get_annotation(args.kwarg.annotation),
                kind="var_keyword",
            ))
        
        return params
    
    def _parse_decorator(self, node: ast.expr) -> Decorator:
        """Parse a decorator."""
        if isinstance(node, ast.Call):
            name = self._get_name(node.func)
            args = [self._get_source(arg) for arg in node.args]
            kwargs = {kw.arg: self._get_source(kw.value) for kw in node.keywords if kw.arg}
            return Decorator(name=name, arguments=args, keyword_arguments=kwargs)
        else:
            return Decorator(name=self._get_name(node))
    
    def _parse_docstring(self, raw: str) -> Docstring:
        """Parse a docstring into structured components."""
        docstring = Docstring(raw=raw)
        
        lines = raw.strip().split('\n')
        if lines:
            docstring.summary = lines[0].strip()
            if len(lines) > 1:
                docstring.description = '\n'.join(lines[1:]).strip()
        
        # Try to use docstring_parser if available
        try:
            from docstring_parser import parse as parse_docstring
            parsed = parse_docstring(raw)
            
            if parsed.short_description:
                docstring.summary = parsed.short_description
            if parsed.long_description:
                docstring.description = parsed.long_description
            
            docstring.params = {
                p.arg_name: p.description or ""
                for p in parsed.params
            }
            
            if parsed.returns and parsed.returns.description:
                docstring.returns = parsed.returns.description
            
            docstring.raises = {
                r.type_name or "Exception": r.description or ""
                for r in parsed.raises
            }
            
            docstring.examples = [
                e.description or ""
                for e in parsed.examples
            ]
        except ImportError:
            pass
        
        return docstring
    
    def _parse_assignment(
        self, 
        node: ast.Assign,
        is_class_var: bool = False
    ) -> list[Variable]:
        """Parse an assignment statement."""
        variables = []
        value_str = self._get_source(node.value)
        
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                variables.append(Variable(
                    name=name,
                    value=value_str,
                    is_constant=name.isupper(),
                    is_class_variable=is_class_var,
                    visibility=self._get_visibility(name),
                    location=self._get_location(node),
                ))
            elif isinstance(target, ast.Tuple):
                # Tuple unpacking
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        variables.append(Variable(
                            name=elt.id,
                            is_class_variable=is_class_var,
                            visibility=self._get_visibility(elt.id),
                            location=self._get_location(node),
                        ))
        
        return variables
    
    def _parse_annotated_assignment(
        self, 
        node: ast.AnnAssign,
        is_class_var: bool = False
    ) -> Optional[Variable]:
        """Parse an annotated assignment."""
        if not isinstance(node.target, ast.Name):
            return None
        
        name = node.target.id
        return Variable(
            name=name,
            type_annotation=self._get_annotation(node.annotation),
            value=self._get_source(node.value) if node.value else None,
            is_constant=name.isupper(),
            is_class_variable=is_class_var,
            visibility=self._get_visibility(name),
            location=self._get_location(node),
        )
    
    def _extract_calls(self, node: ast.AST) -> list[str]:
        """Extract all function calls from a node."""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = self._get_name(child.func)
                if name:
                    calls.append(name)
        
        return list(set(calls))
    
    def _extract_local_variables(
        self, 
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> list[Variable]:
        """Extract local variables from a function."""
        variables = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for var in self._parse_assignment(child):
                    var.is_instance_variable = False
                    variables.append(var)
            elif isinstance(child, ast.AnnAssign):
                var = self._parse_annotated_assignment(child)
                if var:
                    variables.append(var)
        
        return variables
    
    def _is_generator(
        self, 
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> bool:
        """Check if function is a generator."""
        for child in ast.walk(node):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
        return False
    
    def _get_location(self, node: ast.AST) -> CodeLocation:
        """Get location from AST node."""
        return CodeLocation(
            file_path=self.filename,
            start_line=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno) or node.lineno,
            start_col=node.col_offset,
            end_col=getattr(node, 'end_col_offset', 0) or 0,
        )
    
    def _get_name(self, node: ast.expr) -> str:
        """Get name from an expression node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        elif isinstance(node, ast.Subscript):
            return self._get_name(node.value)
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        return ""
    
    def _get_annotation(self, node: Optional[ast.expr]) -> Optional[str]:
        """Get type annotation as string."""
        if node is None:
            return None
        return self._get_source(node)
    
    def _get_source(self, node: Optional[ast.expr]) -> Optional[str]:
        """Get source code for an expression."""
        if node is None:
            return None
        
        try:
            return ast.unparse(node)
        except Exception:
            # Fallback for older Python versions
            if isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Name):
                return node.id
            return None
    
    def _get_visibility(self, name: str) -> Visibility:
        """Determine visibility from name."""
        if name.startswith("__") and not name.endswith("__"):
            return Visibility.PRIVATE
        elif name.startswith("_"):
            return Visibility.PROTECTED
        return Visibility.PUBLIC
