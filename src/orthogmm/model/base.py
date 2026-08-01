from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..types import Array


class BaseMomentModel(ABC):
    """Base contract for models used by OrthoGMM.

    A model must expose unit-level tractable and demanding moment
    contributions. Jacobian and reconstruction methods are optional because
    the estimation layer can construct numerical derivatives and may omit
    post-estimation reconstruction.
    """

    @abstractmethod
    def tractable_moments(self, theta: Array) -> Array:
        """Return an ``n x q_g`` array of tractable moment contributions."""

    @abstractmethod
    def demanding_moments(self, theta: Array) -> Array:
        """Return an ``n x q_h`` array of demanding moment contributions."""

    def tractable_jacobian(self, theta: Array) -> Array:
        """Return the ``q_g x p`` Jacobian of mean tractable moments.

        Subclasses may omit this method entirely when finite differences are
        preferred. Raising ``NotImplementedError`` makes that intent explicit.
        """
        raise NotImplementedError

    def demanding_jacobian(self, theta: Array) -> Array:
        """Return the ``q_h x p`` Jacobian of mean demanding moments."""
        raise NotImplementedError

    def reconstruct(self, theta: Array) -> Any:
        """Optionally reconstruct model-specific quantities at ``theta``."""
        return None

    def unit_ids(self) -> np.ndarray:
        """Optionally return cluster or elementary-unit identifiers."""
        raise NotImplementedError
