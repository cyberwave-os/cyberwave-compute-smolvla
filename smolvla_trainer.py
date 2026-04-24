"""SmolVLA-specific trainer implementation.

Builds TrainPipelineConfig matching the lerobot-train CLI:

    lerobot-train \
        --policy.path=lerobot/smolvla_base \
        --policy.device=cuda \
        --policy.input_features=null \
        --policy.output_features=null \
        --policy.push_to_hub=false \
        --policy.optimizer_lr=1e-3 \
        --policy.scheduler_decay_lr=1e-4 \
        --policy.compile_model=true \
        --policy.compile_mode=reduce-overhead \
        --dataset.repo_id=local/your-dataset \
        --dataset.root=/path/to/dataset \
        --dataset.video_backend=pyav \
        --peft.method_type=LORA \
        --peft.r=16 \
        --output_dir=outputs/train/your-run \
        --steps=50000 \
        --batch_size=32 \
        --num_workers=12 \
        --eval_freq=0 \
        --save_freq=10000 \
        --log_freq=100
"""

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from base_trainer import BaseVLATrainer

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig


def _detect_device() -> str:
    """Auto-detect the best available torch device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SmolVLATrainer(BaseVLATrainer):
    """SmolVLA trainer that builds configs for PEFT/LoRA fine-tuning."""

    MODEL_SLUG = "smolvla"

    def _load_policy_config(self, base_model_path: str) -> Any:
        """Load SmolVLAConfig, handling version mismatches.

        The upstream config.json may include a `type` field that draccus
        doesn't recognize. We work around this by downloading the config,
        stripping unknown meta-fields, and loading from a temp file.
        """
        from draccus.utils import DecodingError
        from huggingface_hub import hf_hub_download
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

        try:
            return SmolVLAConfig.from_pretrained(base_model_path)
        except DecodingError as e:
            if "type" not in str(e):
                raise

            config_file = hf_hub_download(
                repo_id=base_model_path,
                filename="config.json",
            )
            with open(config_file) as f:
                config_data = json.load(f)

            config_data.pop("type", None)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                json.dump(config_data, tmp)
                tmp_path = tmp.name

            try:
                import draccus

                return draccus.parse(SmolVLAConfig, tmp_path, args=[])
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except TypeError as e:
            if "not callable" in str(e):
                raise RuntimeError(
                    "SmolVLA config load failed (lerobot/draccus is incompatible with this "
                    "Python version; use Python 3.12 or 3.13 — see smolvla/install.sh)."
                ) from e
            raise

    def build_pipeline_config(
        self,
        params: dict[str, Any],
        *,
        dataset_root: Path,
        dataset_repo_id: str,
        base_model_path: str,
        output_dir: Path,
    ) -> "TrainPipelineConfig":
        """Build SmolVLA training config.

        Args:
            params: Training parameters including:
                - max_steps: Total training steps (default: 50000)
                - batch_size: Batch size (default: 32)
                - num_workers: DataLoader workers (default: 12)
                - lora_r: LoRA rank (default: 16)
                - optimizer_lr: Learning rate (default: 1e-3)
                - scheduler_decay_lr: LR decay (default: 1e-4)
                - save_freq: Checkpoint frequency (default: 10000)
                - log_freq: Logging frequency (default: 100)
                - compile_model: Use torch.compile (default: True)
                - compile_mode: Compile mode (default: "reduce-overhead")
                - use_peft: Whether to use PEFT/LoRA (default: True)
            dataset_root: Local dataset directory
            dataset_repo_id: Dataset repo ID for lerobot
            base_model_path: HF hub ID or local path
            output_dir: Output directory for checkpoints

        Returns:
            TrainPipelineConfig ready for lerobot train()
        """
        from lerobot.configs.default import DatasetConfig, PeftConfig, WandBConfig
        from lerobot.configs.train import TrainPipelineConfig

        policy_block = params.get("policy")
        max_steps = params.get("max_steps", 50000)
        if isinstance(policy_block, dict) and policy_block.get("max_iterations") is not None:
            max_steps = int(policy_block["max_iterations"])

        policy_cfg = self._load_policy_config(base_model_path)
        policy_cfg.device = params.get("device") or _detect_device()
        policy_cfg.push_to_hub = False
        policy_cfg.input_features = None
        policy_cfg.output_features = None
        policy_cfg.pretrained_path = base_model_path

        policy_cfg.optimizer_lr = params.get("optimizer_lr", 1e-3)
        policy_cfg.scheduler_decay_lr = params.get("scheduler_decay_lr", 1e-4)
        policy_cfg.compile_model = params.get("compile_model", True)
        policy_cfg.compile_mode = params.get("compile_mode", "reduce-overhead")

        use_peft = params.get("use_peft", True)
        peft = None
        if use_peft:
            peft = PeftConfig(
                method_type="LORA",
                r=params.get("lora_r", 16),
            )

        cfg = TrainPipelineConfig(
            dataset=DatasetConfig(
                repo_id=dataset_repo_id,
                root=str(dataset_root),
                video_backend=params.get("video_backend", "pyav"),
            ),
            policy=policy_cfg,
            output_dir=str(output_dir),
            batch_size=params.get("batch_size", 32),
            steps=max_steps,
            eval_freq=params.get("eval_freq", 0),
            save_freq=params.get("save_freq", 10000),
            log_freq=params.get("log_freq", 100),
            num_workers=params.get("num_workers", 12),
            save_checkpoint=True,
            peft=peft,
            wandb=WandBConfig(
                enable=True,
                disable_artifact=True,
            ),
        )

        cfg.validate()
        return cfg
