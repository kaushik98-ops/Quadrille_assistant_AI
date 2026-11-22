"""Shared exception types."""


class AssistantError(Exception):
    """Base class for all project errors."""


class ConfigError(AssistantError):
    """Missing or invalid configuration."""
