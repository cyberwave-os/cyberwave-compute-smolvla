#!/usr/bin/env python
"""SmolVLA training entry point for Cyberwave Cloud Node.

This script is invoked by the Cloud Node with a JSON params file:
    python train.py /path/to/params.json

Expected JSON fields:
    - cyberwave_training_uuid: Training job identifier
    - cyberwave_token: API authentication token (or use CYBERWAVE_API_KEY env)
    - dataset_uuid: UUID of dataset to download
    - dataset_name: Name for the dataset directory
    - data_root_dir: Root directory for datasets (default: ./datasets)
    - base_model: HF hub ID (default: lerobot/smolvla_base)
    - weights_url: Optional URL to custom base weights tar
    - max_steps: Total training steps (default: 50000)
    - batch_size: Batch size (default: 32)
    - lora_r: LoRA rank (default: 16)
    - save_freq: Checkpoint frequency (default: 10000)
    - log_freq: Logging frequency (default: 100)
    - results_folder: Where to put compressed artifacts (default: ./runs/artifacts)

Environment variables:
    - CYBERWAVE_API_KEY: Fallback auth token
    - CYBERWAVE_ENVIRONMENT: "production", "development", or "local"
    - CYBERWAVE_API_URL / CYBERWAVE_BASE_URL: Override API base URL (local: http://localhost:8000)

    On first run, ~/.cyberwave/credentials.json from the Cyberwave CLI is merged into the environment
    (CYBERWAVE_BASE_URL, token) when those vars are unset — use this for local Docker against localhost:8000.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

MODEL_SLUG = "smolvla"


def _setup_logging() -> None:
    """Configure logging with colored output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    """Main entry point for SmolVLA training."""
    from cw_processor import apply_cyberwave_credentials_env

    apply_cyberwave_credentials_env()

    _setup_logging()

    args = argv if argv is not None else sys.argv
    if len(args) < 2:
        print(f"Usage: python {args[0]} /path/to/params.json", file=sys.stderr)
        return 1

    from cw_trainer import CwTrainer, load_json_argument
    from cw_processor import cprint, C

    cprint("\n" + "=" * 60, C.BOLD + C.CYAN)
    cprint("  SmolVLA Training - Cyberwave Cloud Node", C.BOLD + C.CYAN)
    cprint("=" * 60, C.BOLD + C.CYAN)

    try:
        params = load_json_argument(args[1])
    except Exception as e:
        print(f"Error loading params: {e}", file=sys.stderr)
        return 1

    cprint(f"\nReceived {len(params)} parameters", C.WHITE)

    if "params" in params and isinstance(params["params"], dict):
        nested = params.pop("params")
        for key, value in nested.items():
            if key not in params:
                params[key] = value
        cprint(f"Merged {len(nested)} nested params", C.DIM)

    try:
        trainer = CwTrainer(params, model_slug=MODEL_SLUG)
        trainer.setup()
        result = trainer.run()

        cprint("\n" + "=" * 60, C.BOLD + C.GREEN)
        cprint("  Training Complete", C.BOLD + C.GREEN)
        cprint("=" * 60, C.BOLD + C.GREEN)

        print(json.dumps(result))
        return 0

    except KeyboardInterrupt:
        cprint("\nTraining interrupted by user", C.YELLOW)
        return 130
    except Exception as e:
        cprint(f"\nTraining failed: {e}", C.RED)
        logging.exception("Training error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
