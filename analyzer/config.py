# Configuration management

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class ParserConfig:
    max_file_size_mb: int = 10
    supported_extensions: list = field(default_factory=lambda: [".py"])
    encoding: str = "utf-8"
    follow_imports: bool = True


@dataclass
class MetricsConfig:
    calculate_complexity: bool = True
    calculate_halstead: bool = True
    calculate_maintainability: bool = True
    complexity_threshold_warning: int = 10
    complexity_threshold_high: int = 20
    max_function_lines: int = 50
    max_function_params: int = 5


@dataclass
class DependencyConfig:
    analyze_imports: bool = True
    build_call_graph: bool = True
    detect_circular: bool = True
    max_depth: int = 10


@dataclass
class PatternConfig:
    detect_design_patterns: bool = True
    detect_anti_patterns: bool = True
    detect_code_smells: bool = True
    detect_dead_code: bool = True
    detect_duplicates: bool = True
    min_duplicate_lines: int = 5


@dataclass
class SecurityConfig:
    check_sql_injection: bool = True
    check_command_injection: bool = True
    check_hardcoded_secrets: bool = True
    check_dangerous_functions: bool = True
    secret_patterns: list = field(default_factory=list)


@dataclass
class AIConfig:
    max_context_tokens: int = 8000
    include_source_code: bool = True
    summarize_long_files: bool = True
    output_format: str = "json"


@dataclass
class CacheConfig:
    enabled: bool = True
    directory: str = ".analyzer_cache"
    max_size_mb: int = 100
    ttl_hours: int = 24


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    file: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 3


@dataclass
class AnalyzerConfig:
    parser: ParserConfig = field(default_factory=ParserConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    dependencies: DependencyConfig = field(default_factory=DependencyConfig)
    patterns: PatternConfig = field(default_factory=PatternConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    include_patterns: list = field(default_factory=lambda: ["*.py"])
    exclude_patterns: list = field(default_factory=lambda: ["**/venv/**", "**/.git/**", "**/__pycache__/**"])
    max_workers: int = 4
    follow_symlinks: bool = False


class ConfigManager:
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
    
    def load_from_file(self, path: str) -> "ConfigManager":
        path = Path(path)
        if not path.exists():
            return self
        
        content = path.read_text(encoding="utf-8")
        
        if path.suffix in (".yaml", ".yml"):
            if YAML_AVAILABLE:
                data = yaml.safe_load(content)
            else:
                return self
        else:
            data = json.loads(content)
        
        self._apply_config(data)
        return self
    
    def _apply_config(self, data: dict) -> None:
        if not data:
            return
        
        for section, values in data.items():
            if hasattr(self.config, section) and isinstance(values, dict):
                section_config = getattr(self.config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def load_from_env(self) -> "ConfigManager":
        prefix = "CODE_ANALYZER_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                if config_key == "log_level":
                    self.config.logging.level = value
                elif config_key == "cache_enabled":
                    self.config.cache.enabled = value.lower() == "true"
        return self
    
    def save(self, path: str, format: str = "yaml") -> None:
        data = self._to_dict()
        path = Path(path)
        
        if format == "yaml" and YAML_AVAILABLE:
            content = yaml.dump(data, default_flow_style=False)
        else:
            content = json.dumps(data, indent=2)
        
        path.write_text(content, encoding="utf-8")
    
    def _to_dict(self) -> dict:
        result = {}
        for field_name in ["parser", "metrics", "dependencies", "patterns", "security", "ai", "cache", "logging"]:
            section = getattr(self.config, field_name)
            result[field_name] = {k: v for k, v in section.__dict__.items()}
        return result
    
    @classmethod
    def auto_discover(cls, start_path: Optional[Path] = None) -> "ConfigManager":
        manager = cls()
        search_paths = []
        
        if start_path:
            search_paths.append(start_path)
        search_paths.extend([Path.cwd(), Path.home()])
        
        config_names = [".code-analyzer.yaml", ".code-analyzer.yml", ".code-analyzer.json", "code-analyzer.yaml"]
        
        for search_path in search_paths:
            for name in config_names:
                config_file = search_path / name
                if config_file.exists():
                    manager.load_from_file(str(config_file))
                    break
        
        manager.load_from_env()
        return manager


_config: Optional[AnalyzerConfig] = None

def get_config() -> AnalyzerConfig:
    global _config
    if _config is None:
        _config = ConfigManager.auto_discover().config
    return _config

def set_config(config: AnalyzerConfig) -> None:
    global _config
    _config = config
