from typing import List, Any


class HistoryManager:
    """
    Manages calculation history for the advanced calculator.
    """

    def __init__(self) -> None:
        self.history: List[str] = []

    def add_entry(self, entry: str) -> None:
        """Add an entry to the calculation history"""
        self.history.append(entry)

    def clear(self) -> None:
        """Clear calculation history"""
        self.history = []

    def get_entries(self) -> List[str]:
        """Return calculation history"""
        return self.history


def validate_positive_integer(value: Any, param_name: str) -> None:
    """Validate that the input is a positive integer"""
    if not isinstance(value, int):
        raise TypeError(f"{param_name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{param_name} must be positive, got {value}")


def validate_positive_number(value: Any, param_name: str) -> None:
    """Validate that the input is a positive number"""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{param_name} must be a number, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{param_name} must be positive, got {value}")
