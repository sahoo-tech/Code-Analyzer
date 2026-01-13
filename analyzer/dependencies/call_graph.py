"""
Call graph builder.

Builds function/method call graphs from parsed code.
"""

import ast
from pathlib import Path
from typing import Optional

from analyzer.models.code_entities import Module, Function, Method, Class
from analyzer.models.graphs import CallGraph, Node, Edge, NodeType, EdgeType
from analyzer.logging_config import get_logger

logger = get_logger("dependencies.call_graph")


class CallGraphBuilder:
    """
    Builds call graphs from parsed modules.
    
    Creates a graph where:
    - Nodes are functions/methods
    - Edges represent function calls
    """
    
    def __init__(self):
        self.graph = CallGraph(name="call_graph")
        self._module_functions: dict[str, str] = {}  # name -> full_id
    
    def build(self, modules: list[Module]) -> CallGraph:
        """
        Build call graph from multiple modules.
        
        Args:
            modules: List of parsed modules
            
        Returns:
            CallGraph with all function relationships
        """
        # First pass: register all functions
        for module in modules:
            self._register_module(module)
        
        # Second pass: build edges
        for module in modules:
            self._build_edges(module)
        
        return self.graph
    
    def build_for_module(self, module: Module) -> CallGraph:
        """Build call graph for a single module."""
        return self.build([module])
    
    def _register_module(self, module: Module) -> None:
        """Register all functions in a module."""
        module_name = Path(module.file_path).stem if module.file_path else module.name
        
        # Register top-level functions
        for func in module.functions:
            func_id = f"{module_name}.{func.name}"
            self._add_function_node(func, func_id, module.file_path)
            self._module_functions[func.name] = func_id
        
        # Register class methods
        for cls in module.classes:
            self._register_class(cls, module_name, module.file_path)
    
    def _register_class(
        self, 
        cls: Class, 
        module_name: str, 
        file_path: str
    ) -> None:
        """Register all methods in a class."""
        class_id = f"{module_name}.{cls.name}"
        
        # Add class node
        self.graph.add_node(Node(
            id=class_id,
            name=cls.name,
            node_type=NodeType.CLASS,
            file_path=file_path,
            line_number=cls.location.start_line,
        ))
        
        # Register methods
        for method in cls.methods:
            method_id = f"{class_id}.{method.name}"
            self._add_method_node(method, method_id, file_path)
            self._module_functions[f"{cls.name}.{method.name}"] = method_id
        
        # Register nested classes
        for nested in cls.nested_classes:
            self._register_class(nested, class_id, file_path)
    
    def _add_function_node(
        self, 
        func: Function, 
        func_id: str,
        file_path: str
    ) -> None:
        """Add a function node to the graph."""
        self.graph.add_node(Node(
            id=func_id,
            name=func.name,
            node_type=NodeType.FUNCTION,
            file_path=file_path,
            line_number=func.location.start_line,
            metadata={
                "is_async": func.is_async,
                "is_generator": func.is_generator,
                "param_count": len(func.parameters),
            }
        ))
    
    def _add_method_node(
        self, 
        method: Method, 
        method_id: str,
        file_path: str
    ) -> None:
        """Add a method node to the graph."""
        self.graph.add_node(Node(
            id=method_id,
            name=method.name,
            node_type=NodeType.METHOD,
            file_path=file_path,
            line_number=method.location.start_line,
            metadata={
                "is_async": method.is_async,
                "is_static": method.is_static,
                "is_classmethod": method.is_classmethod,
                "is_property": method.is_property,
            }
        ))
    
    def _build_edges(self, module: Module) -> None:
        """Build call edges for a module."""
        module_name = Path(module.file_path).stem if module.file_path else module.name
        
        # Process top-level functions
        for func in module.functions:
            func_id = f"{module_name}.{func.name}"
            self._add_call_edges(func, func_id)
        
        # Process class methods
        for cls in module.classes:
            class_id = f"{module_name}.{cls.name}"
            self._build_class_edges(cls, class_id)
    
    def _build_class_edges(self, cls: Class, class_id: str) -> None:
        """Build edges for class methods."""
        for method in cls.methods:
            method_id = f"{class_id}.{method.name}"
            self._add_call_edges(method, method_id)
        
        for nested in cls.nested_classes:
            nested_id = f"{class_id}.{nested.name}"
            self._build_class_edges(nested, nested_id)
    
    def _add_call_edges(self, func: Function, source_id: str) -> None:
        """Add edges for all calls made by a function."""
        for call_name in func.calls:
            target_id = self._resolve_call(call_name)
            
            if target_id:
                self.graph.add_edge(Edge(
                    source=source_id,
                    target=target_id,
                    edge_type=EdgeType.CALLS,
                ))
            else:
                # External or unresolved call
                external_id = f"external.{call_name}"
                
                if external_id not in self.graph.nodes:
                    self.graph.add_node(Node(
                        id=external_id,
                        name=call_name,
                        node_type=NodeType.EXTERNAL,
                    ))
                
                self.graph.add_edge(Edge(
                    source=source_id,
                    target=external_id,
                    edge_type=EdgeType.CALLS,
                ))
    
    def _resolve_call(self, call_name: str) -> Optional[str]:
        """Resolve a call name to a function ID."""
        # Direct match
        if call_name in self._module_functions:
            return self._module_functions[call_name]
        
        # Try as method call (e.g., "self.method" -> "Class.method")
        if '.' in call_name:
            parts = call_name.split('.')
            if parts[0] in ('self', 'cls'):
                # Need class context - skip for now
                pass
            else:
                # Try direct dotted name
                if call_name in self._module_functions:
                    return self._module_functions[call_name]
        
        return None


def build_call_graph(modules: list[Module]) -> CallGraph:
    """
    Build call graph from modules.
    
    Args:
        modules: List of parsed modules
        
    Returns:
        CallGraph with all function relationships
    """
    builder = CallGraphBuilder()
    return builder.build(modules)
