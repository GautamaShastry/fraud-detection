"""Exception hierarchy for the ML model pipeline."""


class PipelineError(Exception):
    """Base exception for pipeline errors."""


class DataValidationError(PipelineError):
    """Raised when data validation fails."""


class FeatureContractError(PipelineError):
    """Raised when feature contract validation fails."""


class ModelArtifactError(PipelineError):
    """Raised when model artifact is corrupted or invalid."""


class LatencyBudgetExceededError(PipelineError):
    """Raised when inference latency exceeds budget."""


class ApprovalDeniedError(PipelineError):
    """Raised when manual approval is denied."""
