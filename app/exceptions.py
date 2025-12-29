"""
Custom exceptions for the application.
"""


class LegalityAIException(Exception):
    """Base exception for Legality-AI application."""
    pass


class ModelNotLoadedError(LegalityAIException):
    """Raised when required models are not loaded."""
    pass


class InvalidFileError(LegalityAIException):
    """Raised when file validation fails."""
    pass


class FileProcessingError(LegalityAIException):
    """Raised when file processing fails."""
    pass


class ConfigurationError(LegalityAIException):
    """Raised when configuration is invalid."""
    pass

