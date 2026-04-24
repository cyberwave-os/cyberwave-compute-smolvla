from abc import ABC, abstractmethod
from typing import Any


class BaseVLAResolver(ABC):
    MODEL_SLUG: str
    checkpoint: str
    training_config: dict[str, Any] | None
    training_camera_names: list[str]
    expected_state_dim: int
    expected_action_dim: int

    @abstractmethod
    def _load_training_config(self) -> dict[str, Any] | None:
        """Load train_config.json from checkpoint."""
        pass

    @abstractmethod
    def _extract_camera_names(self) -> list[str]:
        """Extract camera names from training config."""
        pass

    @abstractmethod
    def build_camera_mapping(self, runtime_cameras: dict[str, str] | list[str]) -> dict[str, str]:
        """Build camera mapping between training camera names and runtime identifiers."""
        pass

    @abstractmethod
    def get_expected_camera_count(self) -> int:
        """Get expected number of cameras."""
        pass

    @abstractmethod
    def get_expected_state_dim(self) -> int:
        """Get expected state dimension (number of input joints)."""
        pass

    @abstractmethod
    def get_expected_action_dim(self) -> int:
        """Get expected action dimension (number of output joints)."""
        pass
