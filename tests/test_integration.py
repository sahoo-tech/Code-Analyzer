"""Integration tests for the Code Analyzer."""

import pytest
import json
from analyzer import CodeAnalyzer, analyze_code, analyze_file, analyze_directory


class TestCodeAnalyzer:
    """Integration tests for CodeAnalyzer."""
    
    def test_analyze_simple_code(self, sample_code):
        """Test analyzing simple code."""
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(sample_code)
        
        assert result.file_count == 1
        assert len(result.modules) == 1
        assert result.modules[0].name == "<string>"
        
        # Should have parsed entities
        module = result.modules[0]
        assert len(module.functions) >= 2
        assert len(module.classes) >= 1
    
    def test_analyze_file(self, temp_python_file):
        """Test analyzing a file."""
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_file(temp_python_file)
        
        assert result.file_count == 1
        assert len(result.modules) == 1
        assert result.errors == []
    
    def test_analyze_directory(self, temp_project):
        """Test analyzing a directory."""
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_directory(temp_project)
        
        assert result.file_count >= 3
        assert len(result.modules) >= 3
    
    def test_analysis_detects_patterns(self, complex_code):
        """Test that analysis detects patterns."""
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(complex_code)
        
        # Should detect Singleton pattern
        assert len(result.design_patterns) >= 1
        singleton = next((p for p in result.design_patterns if p.pattern_type.value == "singleton"), None)
        assert singleton is not None
    
    def test_analysis_detects_anti_patterns(self, complex_code):
        """Test that analysis detects anti-patterns."""
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(complex_code)
        
        # Should detect God Class
        god_classes = [p for p in result.anti_patterns if p.pattern_type.value == "god_class"]
        assert len(god_classes) >= 1
    
    def test_analysis_detects_security_issues(self, complex_code):
        """Test security vulnerability detection."""
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(complex_code)
        
        # Should detect security issues
        assert len(result.vulnerabilities) >= 1
        
        # Should detect eval
        eval_issues = [v for v in result.vulnerabilities if "eval" in v.message.lower()]
        assert len(eval_issues) >= 1
    
    def test_analysis_detects_secrets(self, complex_code):
        """Test secret detection."""
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(complex_code)
        
        # Should detect hardcoded password
        assert len(result.secrets) >= 1
    
    def test_result_to_json(self, sample_code):
        """Test converting result to JSON."""
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(sample_code)
        
        json_str = result.to_json()
        data = json.loads(json_str)
        
        assert "overview" in data
        assert "modules" in data
        assert "security" in data
    
    def test_get_summary(self, sample_code):
        """Test getting analysis summary."""
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(sample_code)
        
        summary = result.get_summary()
        
        assert "files_analyzed" in summary
        assert "classes" in summary
        assert "functions" in summary
    
    def test_query_interface(self, sample_code):
        """Test natural language query."""
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_code(sample_code)
        
        query_result = analyzer.query(result, "find function greet")
        
        assert "results" in query_result
        assert query_result["count"] >= 1


class TestPublicAPI:
    """Test public API functions."""
    
    def test_analyze_code_function(self, sample_code):
        """Test the analyze_code function."""
        result = analyze_code(sample_code)
        
        assert result.file_count == 1
        assert len(result.modules) == 1
    
    def test_analyze_file_function(self, temp_python_file):
        """Test the analyze_file function."""
        result = analyze_file(temp_python_file)
        
        assert result.file_count == 1
    
    def test_analyze_directory_function(self, temp_project):
        """Test the analyze_directory function."""
        result = analyze_directory(temp_project)
        
        assert result.file_count >= 3
