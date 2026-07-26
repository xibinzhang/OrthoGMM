class OrthoGMMError(Exception):
    """Base exception for the package."""


class ModelContractError(OrthoGMMError):
    """Raised when a model violates the required operator contract."""


class NumericalError(OrthoGMMError):
    """Raised when a numerical operation cannot be completed safely."""
