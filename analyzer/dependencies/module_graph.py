

from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from analyzer.models.code_entities import Module
from analyzer.models.graphs import DependencyGraph, Node, Edge, NodeType, EdgeType
from analyzer.dependencies.import_analyzer import ImportAnalyzer, ImportInfo
from analyzer.logging_config import get_logger

logger = get_logger("dependencies.module_graph")


@dataclass
class ModuleDependency:
    """Information about a module dependency."""
    source_module: str
    target_module: str
    import_count: int = 1
    import_names: list[str] = field(default_factory=list)
    is_circular: bool = False


class ModuleGraphBuilder:
   
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = Path(project_root) if project_root else None
        self.import_analyzer = ImportAnalyzer(project_root)
        self.graph = DependencyGraph(name="module_dependencies")
        self._module_paths: dict[str, str] = {}  # module_id -> file_path
    
    def build(self, modules: list[Module]) -> DependencyGraph:
  
        # Register all modules
        for module in modules:
            self._register_module(module)
        
        # Build dependency edges
        for module in modules:
            self._build_dependencies(module)
        
        return self.graph
    
    def _register_module(self, module: Module) -> None:
        """Register a module in the graph."""
        module_id = self._get_module_id(module)
        
        self.graph.add_node(Node(
            id=module_id,
            name=module.name,
            node_type=NodeType.MODULE,
            file_path=module.file_path,
            metadata={
                "class_count": len(module.classes),
                "function_count": len(module.functions),
                "import_count": len(module.imports),
            }
        ))
        
        self._module_paths[module_id] = module.file_path
    
    def _build_dependencies(self, module: Module) -> None:
        """Build dependency edges for a module."""
        source_id = self._get_module_id(module)
        
        # Analyze imports
        analysis = self.import_analyzer.analyze(module)
        
        # Track dependencies by target module
        dependencies: dict[str, ModuleDependency] = {}
        
        for info in analysis.imports:
            if not info.is_local:
                continue  # Only track local dependencies for now
            
            target_id = self._resolve_import_to_module(info)
            
            if target_id and target_id != source_id:
                if target_id not in dependencies:
                    dependencies[target_id] = ModuleDependency(
                        source_module=source_id,
                        target_module=target_id,
                    )
                else:
                    dependencies[target_id].import_count += 1
                
                # Track imported names
                if info.import_obj.name:
                    dependencies[target_id].import_names.append(info.import_obj.name)
        
        # Add edges
        for target_id, dep in dependencies.items():
            # Add target node if not exists (external local module)
            if target_id not in self.graph.nodes:
                self.graph.add_node(Node(
                    id=target_id,
                    name=Path(target_id).stem,
                    node_type=NodeType.MODULE,
                    file_path=dep.target_module,
                ))
            
            self.graph.add_edge(Edge(
                source=source_id,
                target=target_id,
                edge_type=EdgeType.IMPORTS,
                weight=dep.import_count,
                metadata={
                    "imported_names": dep.import_names,
                }
            ))
    
    def _get_module_id(self, module: Module) -> str:
        """Get unique module ID."""
        if module.file_path:
            # Use relative path from project root if available
            path = Path(module.file_path)
            if self.project_root:
                try:
                    path = path.relative_to(self.project_root)
                except ValueError:
                    pass
            return str(path)
        return module.name
    
    def _resolve_import_to_module(self, info: ImportInfo) -> Optional[str]:
        """Resolve import info to a module ID."""
        if info.resolved_path:
            path = Path(info.resolved_path)
            if self.project_root:
                try:
                    return str(path.relative_to(self.project_root))
                except ValueError:
                    pass
            return str(path)
        return None
    
    def find_circular_dependencies(self) -> list[list[str]]:
   
        return self.graph.find_circular_dependencies()
    
    def get_dependency_order(self) -> list[str]:
      
        return self.graph.topological_sort()
    
    def get_dependency_layers(self) -> list[list[str]]:
    
        return self.graph.get_layers()
    
    def get_module_dependencies(self, module_id: str) -> list[str]:
        """Get modules that a module depends on."""
        deps = self.graph.get_dependencies(module_id)
        return [n.id for n in deps]
    
    def get_module_dependents(self, module_id: str) -> list[str]:
        """Get modules that depend on a module."""
        deps = self.graph.get_dependents(module_id)
        return [n.id for n in deps]


def build_module_graph(
    modules: list[Module],
    project_root: Optional[Path] = None
) -> DependencyGraph:
 
    builder = ModuleGraphBuilder(project_root)
    return builder.build(modules)
