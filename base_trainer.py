"""Abstract base class for VLA model trainers.

Mirrors base_resolver.py for the inference side. Each model type (smolvla, openvla, etc.)
provides a concrete trainer that knows how to build the appropriate TrainPipelineConfig.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


class BaseVLATrainer(ABC):
    """Abstract base class for VLA trainers.

    Each model implementation subclasses this to provide its own config builder.
    The CwTrainer uses a registry to look up the correct trainer by model_slug.
    """

    MODEL_SLUG: str

    @abstractmethod
    def build_pipeline_config(
        self,
        params: dict[str, Any],
        *,
        dataset_root: Path,
        dataset_repo_id: str,
        base_model_path: str,
        output_dir: Path,
    ) -> "TrainPipelineConfig":
        """Build a TrainPipelineConfig for lerobot training.

        Args:
            params: Training parameters from Cyberwave (max_steps, batch_size, lora_r, etc.)
            dataset_root: Local path to the dataset directory
            dataset_repo_id: Dataset repo ID (e.g. "local/dataset-uuid")
            base_model_path: Path or HF hub ID for the base model
            output_dir: Directory to save checkpoints and logs

        Returns:
            A TrainPipelineConfig ready to pass to lerobot's train() function.
        """
        pass
