# Custom exceptions for the analyzer

class AnalyzerError(Exception):
    pass

class ParsingError(AnalyzerError):
    pass

class SyntaxParsingError(ParsingError):
    pass

class SyntaxAnalysisError(ParsingError):
    pass

class UnsupportedLanguageError(ParsingError):
    pass

class FileSystemError(AnalyzerError):
    pass

class FileNotFoundError(FileSystemError):
    pass

class FileReadError(FileSystemError):
    pass

class EncodingError(FileSystemError):
    pass

class ConfigurationError(AnalyzerError):
    pass

class InvalidConfigError(ConfigurationError):
    pass

class MissingConfigError(ConfigurationError):
    pass

class AnalysisError(AnalyzerError):
    pass

class MetricsCalculationError(AnalysisError):
    pass

class DependencyAnalysisError(AnalysisError):
    pass

class PatternDetectionError(AnalysisError):
    pass

class AIIntegrationError(AnalyzerError):
    pass

class ContextTooLargeError(AIIntegrationError):
    pass

class FormattingError(AIIntegrationError):
    pass

class PluginError(AnalyzerError):
    pass

class PluginLoadError(PluginError):
    pass

class PluginExecutionError(PluginError):
    pass

class CacheError(AnalyzerError):
    pass

class CacheReadError(CacheError):
    pass

class CacheWriteError(CacheError):
    pass
