from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseOperator(ABC):
    """Abstract base class for model-independent statistical operators."""

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Construct and return operator-specific statistical objects."""
