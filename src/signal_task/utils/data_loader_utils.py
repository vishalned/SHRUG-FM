"""
Utility functions and classes for data loading operations.

This module provides:
- Enums for data loader types (train/val/test)
- Base class for data loaders
- Helper functions for data loading operations
"""

from enum import Enum


class StrEnum(str, Enum):
    """String enum class that inherits from str and Enum.

    This allows for string comparison while maintaining enum functionality.
    Example:
        >>> DataLoaderType.TRAIN == "train"  # Returns True
        >>> DataLoaderType.TRAIN.value  # Returns "train"
    """

    def __str__(self) -> str:
        return self.value


class DataLoaderType(StrEnum):
    """Enum defining the different types of data loaders.

    Attributes:
        TRAIN: Training data loader
        VAL: Validation data loader
        TEST: Test data loader
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
