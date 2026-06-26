class RewardCompositionError(Exception):
    """Base exception for the public reward composition API."""


class ConfigError(RewardCompositionError):
    """Raised when an experiment or sweep config is invalid."""


class PartialRegistryError(RewardCompositionError):
    """Raised when partial reward registration or loading fails."""


class BackendError(RewardCompositionError):
    """Raised when a backend cannot run or produce expected outputs."""
