"""Test that all modules can be imported correctly."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def _has_torch() -> bool:
    """Check if torch is available."""
    try:
        import torch

        return True
    except ImportError:
        return False


class TestBasicImports:
    """Test basic module imports (no heavy dependencies)."""

    def test_import_cw_processor_utilities(self) -> None:
        """Test importing cw_processor utility functions."""
        from cw_processor import (
            C,
            InferenceRequest,
            cprint,
            load_json_argument,
            normalize_twin_calibration,
            parse_request_payload,
        )

        assert InferenceRequest is not None
        assert callable(parse_request_payload)
        assert callable(load_json_argument)
        assert callable(normalize_twin_calibration)
        assert callable(cprint)
        assert hasattr(C, "RED")
        assert hasattr(C, "GREEN")

    def test_import_base_resolver(self) -> None:
        """Test importing base_resolver."""
        from base_resolver import BaseVLAResolver

        assert BaseVLAResolver is not None
        # MODEL_SLUG is a class variable annotation, verify the class has expected methods
        assert hasattr(BaseVLAResolver, "build_camera_mapping")
        assert hasattr(BaseVLAResolver, "get_expected_camera_count")

    def test_import_smolvla_resolver(self) -> None:
        """Test importing smolvla_resolver."""
        from smolvla_resolver import SmolVLAResolver

        assert SmolVLAResolver is not None
        assert SmolVLAResolver.MODEL_SLUG == "smolvla"

    def test_import_base_trainer(self) -> None:
        """Test importing base_trainer."""
        from base_trainer import BaseVLATrainer

        assert BaseVLATrainer is not None
        # MODEL_SLUG is a class variable annotation, verify the class has expected methods
        assert hasattr(BaseVLATrainer, "build_pipeline_config")

    def test_import_cw_trainer_utilities(self) -> None:
        """Test importing cw_trainer utility functions."""
        from cw_trainer import LogEvent, load_json_argument

        assert LogEvent is not None
        assert callable(load_json_argument)


class TestOptionalImports:
    """Test imports that require optional dependencies."""

    @pytest.mark.skipif(
        not _has_torch(),
        reason="torch not available",
    )
    def test_import_deploy_module(self) -> None:
        """Test importing deploy module (requires torch)."""
        # This import requires torch and lerobot
        try:
            from deploy import main, setup_logging

            assert callable(main)
            assert callable(setup_logging)
        except ImportError as e:
            pytest.skip(f"Deploy dependencies not available: {e}")

    @pytest.mark.skipif(
        not _has_torch(),
        reason="torch not available",
    )
    def test_import_train_module(self) -> None:
        """Test importing train module."""
        try:
            from train import main

            assert callable(main)
        except ImportError as e:
            pytest.skip(f"Train dependencies not available: {e}")
