"""
Secrets detector.

Detects hardcoded secrets and sensitive data:
- API keys
- Passwords
- Tokens
- Private keys
- Connection strings
"""

import re
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from analyzer.models.code_entities import Module
from analyzer.logging_config import get_logger

logger = get_logger("security.secrets")


class SecretType(Enum):
    """Types of secrets."""
    PASSWORD = "password"
    API_KEY = "api_key"
    SECRET_KEY = "secret_key"
    TOKEN = "token"
    PRIVATE_KEY = "private_key"
    CONNECTION_STRING = "connection_string"
    AWS_KEY = "aws_key"
    GITHUB_TOKEN = "github_token"
    GENERIC_SECRET = "generic_secret"


@dataclass
class SecretFinding:
    """A detected secret or sensitive data."""
    secret_type: SecretType
    message: str
    file_path: str
    line_number: int
    variable_name: Optional[str] = None
    matched_pattern: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "type": self.secret_type.value,
            "message": self.message,
            "file": self.file_path,
            "line": self.line_number,
            "variable": self.variable_name,
        }


class SecretsDetector:
    """Detects hardcoded secrets in code."""
    
    # Variable name patterns suggesting secrets
    SECRET_VAR_PATTERNS = [
        (r'(?i)(password|passwd|pwd)', SecretType.PASSWORD),
        (r'(?i)(api_?key|apikey)', SecretType.API_KEY),
        (r'(?i)(secret_?key|secretkey)', SecretType.SECRET_KEY),
        (r'(?i)(auth_?token|authtoken|access_?token)', SecretType.TOKEN),
        (r'(?i)(private_?key|privatekey)', SecretType.PRIVATE_KEY),
        (r'(?i)(connection_?string|conn_?str|database_?url)', SecretType.CONNECTION_STRING),
        (r'(?i)(aws_access_key|aws_secret)', SecretType.AWS_KEY),
        (r'(?i)(github_token|gh_token)', SecretType.GITHUB_TOKEN),
        (r'(?i)(secret|credential)', SecretType.GENERIC_SECRET),
    ]
    
    # Value patterns suggesting secrets
    SECRET_VALUE_PATTERNS = [
        # AWS Access Key
        (r'AKIA[0-9A-Z]{16}', SecretType.AWS_KEY, "AWS Access Key ID"),
        # AWS Secret Key (40 chars, base64-like)
        (r'[A-Za-z0-9/+=]{40}', SecretType.AWS_KEY, "Possible AWS Secret Key"),
        # GitHub Token
        (r'ghp_[A-Za-z0-9]{36}', SecretType.GITHUB_TOKEN, "GitHub Personal Access Token"),
        (r'github_pat_[A-Za-z0-9_]{22,}', SecretType.GITHUB_TOKEN, "GitHub PAT"),
        # Generic API keys (long alphanumeric strings)
        (r'["\'][a-zA-Z0-9]{32,}["\']', SecretType.API_KEY, "Possible API Key"),
        # Private key markers
        (r'-----BEGIN (?:RSA |EC )?PRIVATE KEY-----', SecretType.PRIVATE_KEY, "Private Key"),
        # Connection strings
        (r'(?:mongodb|mysql|postgres|redis)://[^\s"\']+', SecretType.CONNECTION_STRING, "Database Connection String"),
        # JWT tokens
        (r'eyJ[A-Za-z0-9-_]+\.eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+', SecretType.TOKEN, "JWT Token"),
    ]
    
    # Patterns to ignore (false positives)
    IGNORE_PATTERNS = [
        r'(?i)example',
        r'(?i)placeholder',
        r'(?i)your[_-]?',
        r'(?i)test',
        r'(?i)dummy',
        r'(?i)sample',
        r'\*{3,}',  # Masked secrets
        r'<[^>]+>',  # Template placeholders
        r'\$\{[^}]+\}',  # Variable substitution
        r'os\.environ',
        r'os\.getenv',
        r'env\.',
    ]
    
    def __init__(self, min_secret_length: int = 8):
        self.min_secret_length = min_secret_length
    
    def detect(self, modules: list[Module]) -> list[SecretFinding]:
        """
        Detect secrets in modules.
        
        Args:
            modules: List of parsed modules
            
        Returns:
            List of secret findings
        """
        findings = []
        
        for module in modules:
            findings.extend(self._scan_module(module))
        
        return findings
    
    def detect_from_code(self, code: str, file_path: str = "<string>") -> list[SecretFinding]:
        """Detect secrets in code string."""
        findings = []
        
        for i, line in enumerate(code.splitlines(), 1):
            findings.extend(self._scan_line(line, file_path, i))
        
        return findings
    
    def _scan_module(self, module: Module) -> list[SecretFinding]:
        """Scan a module for secrets."""
        findings = []
        
        # Check module-level variables
        for var in module.variables + module.constants:
            finding = self._check_variable(var, module.file_path)
            if finding:
                findings.append(finding)
        
        # Check class variables
        for cls in module.classes:
            for var in cls.class_variables:
                finding = self._check_variable(var, module.file_path)
                if finding:
                    findings.append(finding)
        
        return findings
    
    def _check_variable(self, var, file_path: str) -> Optional[SecretFinding]:
        """Check a variable for secret patterns."""
        line_number = var.location.start_line if var.location else 1
        
        # Check variable name
        for pattern, secret_type in self.SECRET_VAR_PATTERNS:
            if re.search(pattern, var.name):
                # Check if value looks like a hardcoded secret
                if var.value and self._is_hardcoded_secret(var.value):
                    return SecretFinding(
                        secret_type=secret_type,
                        message=f"Possible hardcoded {secret_type.value} in variable '{var.name}'",
                        file_path=file_path,
                        line_number=line_number,
                        variable_name=var.name,
                        matched_pattern=pattern,
                    )
        
        # Check value patterns regardless of variable name
        if var.value:
            for pattern, secret_type, message in self.SECRET_VALUE_PATTERNS:
                if re.search(pattern, var.value):
                    if not self._is_ignored(var.value):
                        return SecretFinding(
                            secret_type=secret_type,
                            message=f"{message} found",
                            file_path=file_path,
                            line_number=line_number,
                            variable_name=var.name,
                            matched_pattern=pattern,
                        )
        
        return None
    
    def _scan_line(self, line: str, file_path: str, line_number: int) -> list[SecretFinding]:
        """Scan a single line for secrets."""
        findings = []
        
        if self._is_ignored(line):
            return findings
        
        # Check value patterns
        for pattern, secret_type, message in self.SECRET_VALUE_PATTERNS:
            if re.search(pattern, line):
                # Extract variable name if assignment
                var_match = re.match(r'\s*(\w+)\s*=', line)
                var_name = var_match.group(1) if var_match else None
                
                findings.append(SecretFinding(
                    secret_type=secret_type,
                    message=f"{message} found",
                    file_path=file_path,
                    line_number=line_number,
                    variable_name=var_name,
                    matched_pattern=pattern,
                ))
        
        # Check variable name patterns with string assignments
        assignment_match = re.match(r'\s*(\w+)\s*=\s*["\']([^"\']+)["\']', line)
        if assignment_match:
            var_name = assignment_match.group(1)
            value = assignment_match.group(2)
            
            for pattern, secret_type in self.SECRET_VAR_PATTERNS:
                if re.search(pattern, var_name):
                    if self._is_hardcoded_secret(value):
                        findings.append(SecretFinding(
                            secret_type=secret_type,
                            message=f"Hardcoded {secret_type.value}",
                            file_path=file_path,
                            line_number=line_number,
                            variable_name=var_name,
                        ))
                    break
        
        return findings
    
    def _is_hardcoded_secret(self, value: str) -> bool:
        """Check if a value looks like a hardcoded secret."""
        # Remove quotes
        value = value.strip("'\"")
        
        # Too short
        if len(value) < self.min_secret_length:
            return False
        
        # Check if it's a placeholder
        if self._is_ignored(value):
            return False
        
        # Environment variable reference
        if value.startswith(('os.environ', 'os.getenv', 'env.')):
            return False
        
        # Empty or None
        if value in ('None', 'null', '""', "''", ''):
            return False
        
        return True
    
    def _is_ignored(self, value: str) -> bool:
        """Check if value should be ignored (likely false positive)."""
        for pattern in self.IGNORE_PATTERNS:
            if re.search(pattern, value):
                return True
        return False


def detect_secrets(modules: list[Module]) -> list[SecretFinding]:
    """
    Detect secrets in modules.
    
    Args:
        modules: List of parsed modules
        
    Returns:
        List of secret findings
    """
    detector = SecretsDetector()
    return detector.detect(modules)
