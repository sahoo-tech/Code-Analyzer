

from dataclasses import dataclass, field
from typing import Optional, Any, Iterator
from enum import Enum


class EdgeType(Enum):
    """Types of edges in graphs."""
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    USES = "uses"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"


class NodeType(Enum):
    """Types of nodes in graphs."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    PACKAGE = "package"
    EXTERNAL = "external"


@dataclass
class Node:
    """Node in a graph."""
    id: str
    name: str
    node_type: NodeType
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash(self.id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.id == other.id
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.node_type.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "metadata": self.metadata,
        }


@dataclass
class Edge:
    """Edge in a graph."""
    source: str  # Node ID
    target: str  # Node ID
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self) -> int:
        return hash((self.source, self.target, self.edge_type))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False
        return (
            self.source == other.source and
            self.target == other.target and
            self.edge_type == other.edge_type
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }


@dataclass
class Graph:
    """Base graph structure."""
    name: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        if edge not in self.edges:
            self.edges.append(edge)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_successors(self, node_id: str) -> list[Node]:
        """Get nodes that this node points to."""
        return [
            self.nodes[e.target]
            for e in self.edges
            if e.source == node_id and e.target in self.nodes
        ]
    
    def get_predecessors(self, node_id: str) -> list[Node]:
        """Get nodes that point to this node."""
        return [
            self.nodes[e.source]
            for e in self.edges
            if e.target == node_id and e.source in self.nodes
        ]
    
    def get_edges_from(self, node_id: str) -> list[Edge]:
        """Get all edges from a node."""
        return [e for e in self.edges if e.source == node_id]
    
    def get_edges_to(self, node_id: str) -> list[Edge]:
        """Get all edges to a node."""
        return [e for e in self.edges if e.target == node_id]
    
    @property
    def node_count(self) -> int:
        return len(self.nodes)
    
    @property
    def edge_count(self) -> int:
        return len(self.edges)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }


@dataclass
class CallGraph(Graph):
    """Function/method call graph."""
    
    def get_callers(self, function_id: str) -> list[Node]:
        """Get functions that call this function."""
        return self.get_predecessors(function_id)
    
    def get_callees(self, function_id: str) -> list[Node]:
        """Get functions called by this function."""
        return self.get_successors(function_id)
    
    def get_call_depth(self, function_id: str, visited: Optional[set] = None) -> int:
        """Get maximum call depth from a function."""
        if visited is None:
            visited = set()
        
        if function_id in visited:
            return 0  # Cycle detected
        
        visited.add(function_id)
        callees = self.get_callees(function_id)
        
        if not callees:
            return 0
        
        max_depth = 0
        for callee in callees:
            depth = self.get_call_depth(callee.id, visited.copy())
            max_depth = max(max_depth, depth + 1)
        
        return max_depth
    
    def find_recursive_calls(self) -> list[str]:
        """Find functions that call themselves (directly or indirectly)."""
        recursive = []
        
        for node_id in self.nodes:
            if self._has_path_to_self(node_id):
                recursive.append(node_id)
        
        return recursive
    
    def _has_path_to_self(self, start_id: str) -> bool:
        """Check if there's a path from node back to itself."""
        visited = set()
        stack = [start_id]
        first = True
        
        while stack:
            current = stack.pop()
            
            if current == start_id and not first:
                return True
            
            first = False
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for edge in self.get_edges_from(current):
                if edge.target not in visited or edge.target == start_id:
                    stack.append(edge.target)
        
        return False


@dataclass
class DependencyGraph(Graph):
    """Module/package dependency graph."""
    
    def get_dependencies(self, module_id: str) -> list[Node]:
        """Get modules that this module depends on."""
        return self.get_successors(module_id)
    
    def get_dependents(self, module_id: str) -> list[Node]:
        """Get modules that depend on this module."""
        return self.get_predecessors(module_id)
    
    def find_circular_dependencies(self) -> list[list[str]]:
        """Find all circular dependency cycles."""
        cycles = []
        visited = set()
        rec_stack: dict[str, int] = {}
        path: list[str] = []
        
        def dfs(node_id: str) -> None:
            visited.add(node_id)
            rec_stack[node_id] = len(path)
            path.append(node_id)
            
            for edge in self.get_edges_from(node_id):
                target = edge.target
                
                if target not in visited:
                    dfs(target)
                elif target in rec_stack:
                    # Found cycle
                    cycle_start = rec_stack[target]
                    cycle = path[cycle_start:] + [target]
                    cycles.append(cycle)
            
            path.pop()
            del rec_stack[node_id]
        
        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id)
        
        return cycles
    
    def get_dependency_depth(self, module_id: str) -> int:
        """Get the depth of dependencies for a module."""
        visited = set()
        
        def dfs(node_id: str) -> int:
            if node_id in visited:
                return 0
            
            visited.add(node_id)
            deps = self.get_dependencies(node_id)
            
            if not deps:
                return 0
            
            return 1 + max(dfs(d.id) for d in deps)
        
        return dfs(module_id)
    
    def topological_sort(self) -> list[str]:
        """Return nodes in topological order (dependencies first)."""
        in_degree = {node_id: 0 for node_id in self.nodes}
        
        for edge in self.edges:
            if edge.target in in_degree:
                in_degree[edge.target] += 1
        
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            for edge in self.get_edges_from(node_id):
                if edge.target in in_degree:
                    in_degree[edge.target] -= 1
                    if in_degree[edge.target] == 0:
                        queue.append(edge.target)
        
        # Check for cycles
        if len(result) != len(self.nodes):
            # Graph has cycles, return partial order
            remaining = [n for n in self.nodes if n not in result]
            result.extend(remaining)
        
        return result
    
    def get_layers(self) -> list[list[str]]:
 
        layers: dict[int, list[str]] = {}
        
        for node_id in self.nodes:
            depth = self.get_dependency_depth(node_id)
            if depth not in layers:
                layers[depth] = []
            layers[depth].append(node_id)
        
        return [layers[i] for i in sorted(layers.keys())]


@dataclass
class InheritanceGraph(Graph):
    """Class inheritance graph."""
    
    def get_ancestors(self, class_id: str) -> list[Node]:
        """Get all ancestor classes (transitive)."""
        ancestors = []
        visited = set()
        stack = [class_id]
        
        while stack:
            current = stack.pop()
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for parent in self.get_successors(current):
                if parent.id not in visited:
                    ancestors.append(parent)
                    stack.append(parent.id)
        
        return ancestors
    
    def get_descendants(self, class_id: str) -> list[Node]:
        """Get all descendant classes (transitive)."""
        descendants = []
        visited = set()
        stack = [class_id]
        
        while stack:
            current = stack.pop()
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for child in self.get_predecessors(current):
                if child.id not in visited:
                    descendants.append(child)
                    stack.append(child.id)
        
        return descendants
    
    def get_depth(self, class_id: str) -> int:
        """Get inheritance depth of a class."""
        parents = self.get_successors(class_id)
        
        if not parents:
            return 0
        
        return 1 + max(self.get_depth(p.id) for p in parents)
    
    def find_diamond_inheritance(self) -> list[tuple[str, list[str]]]:
        """Find diamond inheritance patterns."""
        diamonds = []
        
        for node_id in self.nodes:
            parents = self.get_successors(node_id)
            
            if len(parents) < 2:
                continue
            
            # Check if parents share common ancestors
            ancestor_sets = [
                set(a.id for a in self.get_ancestors(p.id))
                for p in parents
            ]
            
            if len(ancestor_sets) >= 2:
                common = ancestor_sets[0]
                for s in ancestor_sets[1:]:
                    common = common.intersection(s)
                
                if common:
                    diamonds.append((node_id, list(common)))
        
        return diamonds
