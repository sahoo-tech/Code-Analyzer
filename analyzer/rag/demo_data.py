"""Demo data and sample project for RAG demonstrations.

Creates a sample Python project with various code entities that can be
indexed and queried through the RAG system.
"""

import os
from pathlib import Path
from typing import Optional, List


# Sample Python files that will be created for demo
DEMO_FILES = {
    "calculator.py": '''"""A simple calculator module for demonstration.

This module provides basic arithmetic operations and an advanced
calculator class with memory functionality.
"""

from typing import Optional, List
from dataclasses import dataclass


@dataclass
class CalculationResult:
    """Represents the result of a calculation."""
    value: float
    operation: str
    operands: tuple
    
    def __str__(self) -> str:
        return f"{self.operation}: {self.operands} = {self.value}"


def add(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
        
    Example:
        >>> add(2, 3)
        5
    """
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> Optional[float]:
    """Safely divide a by b.
    
    Returns None if b is zero to avoid division errors.
    """
    if b == 0:
        return None
    return a / b


class Calculator:
    """An advanced calculator with memory functionality.
    
    This calculator maintains a history of operations and supports
    memory storage for intermediate results.
    
    Attributes:
        history: List of past calculation results
        memory: Stored value for recall
    """
    
    def __init__(self):
        self.history: List[CalculationResult] = []
        self.memory: float = 0.0
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers and store in history."""
        result = add(a, b)
        self._record(result, "add", (a, b))
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract and store in history."""
        result = subtract(a, b)
        self._record(result, "subtract", (a, b))
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply and store in history."""
        result = multiply(a, b)
        self._record(result, "multiply", (a, b))
        return result
    
    def divide(self, a: float, b: float) -> Optional[float]:
        """Divide and store in history."""
        result = divide(a, b)
        if result is not None:
            self._record(result, "divide", (a, b))
        return result
    
    def memory_store(self, value: float) -> None:
        """Store a value in memory."""
        self.memory = value
    
    def memory_recall(self) -> float:
        """Recall the stored value."""
        return self.memory
    
    def memory_clear(self) -> None:
        """Clear the memory."""
        self.memory = 0.0
    
    def get_history(self) -> List[CalculationResult]:
        """Get all calculation history."""
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Clear the calculation history."""
        self.history.clear()
    
    def _record(self, value: float, operation: str, operands: tuple) -> None:
        """Record a calculation in history."""
        result = CalculationResult(value, operation, operands)
        self.history.append(result)
''',

    "user_management.py": '''"""User management system for demonstrations.

Provides user authentication, authorization, and profile management.
"""

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum


class UserRole(Enum):
    """User roles for authorization."""
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"
    SUPERADMIN = "superadmin"


@dataclass
class User:
    """Represents a system user.
    
    Attributes:
        username: Unique identifier for the user
        email: User email address
        role: User role for permissions
        created_at: Account creation timestamp
    """
    username: str
    email: str
    role: UserRole = UserRole.USER
    created_at: datetime = field(default_factory=datetime.now)
    password_hash: str = ""
    is_active: bool = True
    
    def has_permission(self, required_role: UserRole) -> bool:
        """Check if user has at least the required role level."""
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.USER: 1,
            UserRole.ADMIN: 2,
            UserRole.SUPERADMIN: 3,
        }
        return role_hierarchy[self.role] >= role_hierarchy[required_role]


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthorizationError(Exception):
    """Raised when user lacks required permissions."""
    pass


class UserManager:
    """Manages user accounts and authentication.
    
    This class handles user creation, authentication, and authorization
    for the application.
    """
    
    def __init__(self):
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, str] = {}  # token -> username
    
    def create_user(
        self, 
        username: str, 
        email: str, 
        password: str,
        role: UserRole = UserRole.USER
    ) -> User:
        """Create a new user account.
        
        Args:
            username: Unique username
            email: User email
            password: Plain text password (will be hashed)
            role: User role (default: USER)
            
        Returns:
            Created User object
            
        Raises:
            ValueError: If username already exists
        """
        if username in self._users:
            raise ValueError(f"Username '{username}' already exists")
        
        user = User(
            username=username,
            email=email,
            role=role,
            password_hash=self._hash_password(password),
        )
        self._users[username] = user
        return user
    
    def authenticate(self, username: str, password: str) -> str:
        """Authenticate a user and return a session token.
        
        Args:
            username: User username
            password: User password
            
        Returns:
            Session token for authenticated session
            
        Raises:
            AuthenticationError: If credentials are invalid
        """
        user = self._users.get(username)
        
        if not user:
            raise AuthenticationError("Invalid credentials")
        
        if not user.is_active:
            raise AuthenticationError("Account is disabled")
        
        if user.password_hash != self._hash_password(password):
            raise AuthenticationError("Invalid credentials")
        
        token = secrets.token_urlsafe(32)
        self._sessions[token] = username
        return token
    
    def logout(self, token: str) -> bool:
        """Log out a user session."""
        if token in self._sessions:
            del self._sessions[token]
            return True
        return False
    
    def get_current_user(self, token: str) -> Optional[User]:
        """Get the user for a session token."""
        username = self._sessions.get(token)
        if username:
            return self._users.get(username)
        return None
    
    def require_role(self, token: str, required_role: UserRole) -> User:
        """Verify user has required role.
        
        Args:
            token: Session token
            required_role: Minimum required role
            
        Returns:
            User object if authorized
            
        Raises:
            AuthorizationError: If user lacks permission
        """
        user = self.get_current_user(token)
        if not user:
            raise AuthorizationError("Not authenticated")
        
        if not user.has_permission(required_role):
            raise AuthorizationError(
                f"Requires {required_role.value} role"
            )
        
        return user
    
    def list_users(self) -> List[User]:
        """Get all users."""
        return list(self._users.values())
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate a user account."""
        user = self._users.get(username)
        if user:
            user.is_active = False
            return True
        return False
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """Hash a password for secure storage."""
        return hashlib.sha256(password.encode()).hexdigest()
''',

    "data_processor.py": '''"""Data processing utilities for ETL operations.

Provides functions for loading, transforming, and validating data.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ProcessingResult:
    """Result of a data processing operation."""
    success: bool
    records_processed: int
    errors: List[str]
    output_path: Optional[str] = None


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, path: str) -> List[Dict[str, Any]]:
        """Load data from a source."""
        pass


class JSONLoader(DataLoader):
    """Loads data from JSON files."""
    
    def load(self, path: str) -> List[Dict[str, Any]]:
        """Load JSON data from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        return [data]


class CSVLoader(DataLoader):
    """Loads data from CSV files."""
    
    def load(self, path: str) -> List[Dict[str, Any]]:
        """Load CSV data from file."""
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records


class DataTransformer:
    """Transforms data records with configurable operations."""
    
    def __init__(self):
        self._transformations: List[Callable] = []
    
    def add_transform(self, func: Callable) -> 'DataTransformer':
        """Add a transformation function."""
        self._transformations.append(func)
        return self
    
    def transform(self, records: List[Dict]) -> List[Dict]:
        """Apply all transformations to records."""
        result = records.copy()
        for transform in self._transformations:
            result = [transform(r) for r in result]
        return result


class DataValidator:
    """Validates data records against rules."""
    
    def __init__(self):
        self._rules: List[Callable[[Dict], Optional[str]]] = []
    
    def add_rule(
        self, 
        rule: Callable[[Dict], Optional[str]]
    ) -> 'DataValidator':
        """Add a validation rule.
        
        Rule should return None if valid, error message otherwise.
        """
        self._rules.append(rule)
        return self
    
    def validate(self, records: List[Dict]) -> List[str]:
        """Validate all records and return error messages."""
        errors = []
        for i, record in enumerate(records):
            for rule in self._rules:
                error = rule(record)
                if error:
                    errors.append(f"Record {i}: {error}")
        return errors


class DataPipeline:
    """Orchestrates data loading, transformation, and validation."""
    
    def __init__(
        self,
        loader: DataLoader,
        transformer: Optional[DataTransformer] = None,
        validator: Optional[DataValidator] = None,
    ):
        self.loader = loader
        self.transformer = transformer
        self.validator = validator
    
    def process(self, input_path: str) -> ProcessingResult:
        """Run the full data processing pipeline."""
        errors = []
        
        # Load data
        try:
            records = self.loader.load(input_path)
        except Exception as e:
            return ProcessingResult(
                success=False,
                records_processed=0,
                errors=[f"Load error: {e}"],
            )
        
        # Transform
        if self.transformer:
            records = self.transformer.transform(records)
        
        # Validate
        if self.validator:
            validation_errors = self.validator.validate(records)
            errors.extend(validation_errors)
        
        return ProcessingResult(
            success=len(errors) == 0,
            records_processed=len(records),
            errors=errors,
        )
'''
}


DEMO_QUESTIONS = [
    "How does the Calculator class store history?",
    "What is the authentication flow for users?",
    "How do I add two numbers together?",
    "What roles are available for users?",
    "How does the data pipeline work?",
    "What validation is performed on data?",
    "How do I create a new user account?",
    "What exceptions can be raised during authentication?",
]


def create_demo_project(base_path: Optional[str] = None) -> Path:
    """Create a demo project for RAG testing.
    
    Args:
        base_path: Optional base path. Defaults to .analyzer_demo
        
    Returns:
        Path to created demo project
    """
    if base_path:
        demo_dir = Path(base_path)
    else:
        demo_dir = Path(".analyzer_demo")
    
    demo_dir.mkdir(exist_ok=True)
    
    # Create demo files
    for filename, content in DEMO_FILES.items():
        file_path = demo_dir / filename
        file_path.write_text(content, encoding='utf-8')
    
    # Create __init__.py
    init_content = '''"""Demo project for Code Analyzer RAG demonstrations."""

from .calculator import Calculator, add, subtract, multiply, divide
from .user_management import UserManager, User, UserRole
from .data_processor import DataPipeline, JSONLoader, CSVLoader
'''
    (demo_dir / "__init__.py").write_text(init_content, encoding='utf-8')
    
    return demo_dir


def get_demo_questions() -> List[str]:
    """Get list of demo questions for RAG testing."""
    return DEMO_QUESTIONS.copy()


def cleanup_demo_project(path: Optional[Path] = None) -> bool:
    """Clean up demo project files.
    
    Args:
        path: Path to demo project. Defaults to .analyzer_demo
        
    Returns:
        True if cleanup successful
    """
    import shutil
    
    if path is None:
        path = Path(".analyzer_demo")
    
    if path.exists():
        shutil.rmtree(path)
        return True
    return False
