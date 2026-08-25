"""Tests for deploy.py camera reconciliation and checkpoint config loading.

Context: a fine-tuned SmolVLA checkpoint declares its own camera names in
``config.json`` (e.g. ``wrist``/``top``), while ``lerobot/smolvla_base`` declares
``camera1``/``camera2``/``camera3``. Building the policy from the base config
served the wrong feature contract and made inference fail with a hard KeyError
on the first control cycle. These tests pin the checkpoint as the source of
truth and pin best-effort behaviour when the runtime frames don't line up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402

import deploy  # noqa: E402


def _cfg(camera_names: list[str], *, state_dim: int = 6, action_dim: int = 6):
    """Minimal stand-in for SmolVLAConfig with the bits deploy.py reads."""

    class _Cfg:
        input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            **{
                f"observation.images.{n}": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 480, 640)
                )
                for n in camera_names
            },
        }
        output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
        }

    return _Cfg()


def _img(tag: int) -> np.ndarray:
    return np.full((480, 640, 3), tag, dtype=np.uint8)


class TestResolveCameraInputs:
    """deploy.resolve_camera_inputs: map runtime frames onto declared cameras."""

    def test_exact_name_match_is_preferred(self) -> None:
        cfg = _cfg(["wrist", "top"])
        images = {"top": _img(1), "wrist": _img(2)}

        mapped, missing = deploy.resolve_camera_inputs(cfg, images)

        assert missing == []
        # Matched by name, not by the order the runtime happened to send them.
        assert int(mapped["wrist"][0, 0, 0]) == 2
        assert int(mapped["top"][0, 0, 0]) == 1

    def test_positional_fallback_when_names_do_not_match(self) -> None:
        """Execution must proceed even when the runtime uses unrelated names."""
        cfg = _cfg(["wrist", "top"])
        images = {"primary_camera": _img(1), "wrist_camera": _img(2)}

        mapped, missing = deploy.resolve_camera_inputs(cfg, images)

        assert missing == []
        assert sorted(mapped) == ["top", "wrist"]
        assert int(mapped["wrist"][0, 0, 0]) == 1
        assert int(mapped["top"][0, 0, 0]) == 2

    def test_partial_name_match_fills_remainder_positionally(self) -> None:
        cfg = _cfg(["wrist", "top"])
        images = {"top": _img(9), "some_other_cam": _img(4)}

        mapped, missing = deploy.resolve_camera_inputs(cfg, images)

        assert missing == []
        assert int(mapped["top"][0, 0, 0]) == 9, "named match must not be stolen"
        assert int(mapped["wrist"][0, 0, 0]) == 4

    def test_fewer_frames_than_cameras_reports_missing(self) -> None:
        cfg = _cfg(["wrist", "top"])

        mapped, missing = deploy.resolve_camera_inputs(cfg, {"top": _img(1)})

        assert sorted(mapped) == ["top"]
        assert missing == ["wrist"]

    def test_extra_runtime_frames_are_ignored(self) -> None:
        cfg = _cfg(["wrist"])
        images = {"wrist": _img(1), "spare": _img(2), "spare2": _img(3)}

        mapped, missing = deploy.resolve_camera_inputs(cfg, images)

        assert sorted(mapped) == ["wrist"]
        assert missing == []

    def test_no_frames_at_all(self) -> None:
        cfg = _cfg(["wrist", "top"])

        mapped, missing = deploy.resolve_camera_inputs(cfg, {})

        assert mapped == {}
        assert sorted(missing) == ["top", "wrist"]

    def test_controller_slots_take_precedence_over_checkpoint_order(self) -> None:
        images = {"top": _img(1), "wrist": _img(2)}

        contract_images, missing = deploy.resolve_named_camera_inputs(
            ["top", "wrist"], images
        )
        mapped, model_missing = deploy.map_camera_inputs_positionally(
            ["wrist", "top"], contract_images
        )

        assert missing == []
        assert model_missing == []
        assert int(mapped["wrist"][0, 0, 0]) == 1
        assert int(mapped["top"][0, 0, 0]) == 2


class TestRawObservationBestEffort:
    """A missing camera must not abort the control cycle."""

    def test_missing_camera_is_skipped_not_raised(self) -> None:
        cfg = _cfg(["wrist", "top"])
        ds_features = deploy.dataset_features_from_policy(cfg)

        raw = deploy.raw_observation_from_tensors(
            cfg, ds_features, {"top": _img(1)}, np.zeros(6, dtype=np.float32)
        )

        assert "top" in raw
        assert "wrist" not in raw
        assert raw["state_0"] == 0.0

    def test_all_cameras_present_unchanged(self) -> None:
        cfg = _cfg(["wrist", "top"])
        ds_features = deploy.dataset_features_from_policy(cfg)

        raw = deploy.raw_observation_from_tensors(
            cfg, ds_features, {"top": _img(1), "wrist": _img(2)}, np.zeros(6, dtype=np.float32)
        )

        assert {"top", "wrist"} <= set(raw)


class TestFrameFeaturesForAvailable:
    """ds_features must drop absent cameras or lerobot's frame builder KeyErrors."""

    def test_absent_cameras_removed(self) -> None:
        cfg = _cfg(["wrist", "top"])
        ds_features = deploy.dataset_features_from_policy(cfg)

        filtered = deploy.frame_features_for_available(ds_features, ["wrist"])

        assert "observation.images.top" in filtered
        assert "observation.images.wrist" not in filtered
        assert "observation.state" in filtered
        assert "action" in filtered

    def test_nothing_missing_returns_equivalent_mapping(self) -> None:
        cfg = _cfg(["wrist", "top"])
        ds_features = deploy.dataset_features_from_policy(cfg)

        assert deploy.frame_features_for_available(ds_features, []) == ds_features


class TestLoadCheckpointPolicyConfig:
    """The fine-tuned checkpoint - not the base model - defines the contract."""

    def _write_ckpt(self, tmp_path: Path, extra: dict | None = None) -> Path:
        cfg = {
            "type": "smolvla",
            "chunk_size": 50,
            "n_action_steps": 50,
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]},
                "observation.images.top": {"type": "VISUAL", "shape": [3, 480, 640]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
            "normalization_mapping": {
                "VISUAL": "IDENTITY",
                "STATE": "MEAN_STD",
                "ACTION": "MEAN_STD",
            },
        }
        cfg.update(extra or {})
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        return tmp_path

    def test_returns_checkpoint_cameras(self, tmp_path: Path) -> None:
        ckpt = self._write_ckpt(tmp_path)

        cfg = deploy.load_checkpoint_policy_config(str(ckpt), torch.device("cpu"))

        assert cfg is not None
        cams = [
            k.split("observation.images.")[-1]
            for k, v in cfg.input_features.items()
            if v.type == FeatureType.VISUAL
        ]
        assert sorted(cams) == ["top", "wrist"], "must not fall back to camera1/2/3"

    def test_compile_model_disabled_before_construction(self, tmp_path: Path) -> None:
        """Training checkpoints carry compile_model=true; the policy compiles in
        __init__, so clearing it after from_pretrained is too late."""
        ckpt = self._write_ckpt(
            tmp_path, {"compile_model": True, "compile_mode": "reduce-overhead"}
        )

        cfg = deploy.load_checkpoint_policy_config(str(ckpt), torch.device("cpu"))

        assert cfg is not None, "unknown newer-lerobot fields must not defeat parsing"
        assert cfg.compile_model is False

    def test_missing_config_returns_none(self, tmp_path: Path) -> None:
        assert deploy.load_checkpoint_policy_config(str(tmp_path), torch.device("cpu")) is None

    def test_unparseable_config_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{not json")

        assert deploy.load_checkpoint_policy_config(str(tmp_path), torch.device("cpu")) is None
