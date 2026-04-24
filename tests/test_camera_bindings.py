"""Tests for background camera frame fetcher functionality."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


VALID_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    + b"\xff\xdb\x00C" + b"\x10" * 64
    + b"\xff\xc0\x00\x0b\x08\x00\x02\x00\x02\x01\x01\x11\x00"
    + b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    + b"\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    + b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\x7f\x00\x7f\xff\xd9"
)


@pytest.fixture
def small_valid_jpeg() -> bytes:
    """A minimal valid JPEG image (2x2 pixels)."""
    from PIL import Image
    import io
    img = Image.new("RGB", (4, 4), color=(128, 64, 192))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestCameraBindings:
    """Tests for camera binding and background fetcher functionality."""

    def test_camera_binding_created_for_mapped_cameras(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
        small_valid_jpeg: bytes,
    ) -> None:
        """Test that CameraBinding objects are created for each mapped camera."""
        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        mock_client = MagicMock()
        mock_camera_twin = MagicMock()
        mock_camera_twin.get_latest_frame.return_value = small_valid_jpeg
        mock_client.twins.get.return_value = mock_camera_twin
        mock_client.mqtt.connect.return_value = None
        mock_client.mqtt.subscribe_joint_states.return_value = None
        mock_client.affect.return_value = mock_client

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=mock_client,
        )

        processor.client = mock_client
        processor.camera_mapping = {
            "observation.images.cam_abc123": "primary_camera",
            "observation.images.cam_def456": "wrist_camera",
        }

        processor._setup_camera_bindings()

        assert len(processor.cameras) == 2
        assert "observation.images.cam_abc123" in processor.cameras
        assert "observation.images.cam_def456" in processor.cameras

        processor.disconnect()

    def test_get_inputs_reads_from_cached_images(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
        small_valid_jpeg: bytes,
    ) -> None:
        """Test that get_inputs reads from cached np.ndarray, not REST."""
        from cw_processor import CwProcessor, CameraBinding, parse_request_payload
        from PIL import Image
        import io

        request = parse_request_payload(json.dumps(sample_inference_payload))

        mock_client = MagicMock()
        mock_client.affect.return_value = mock_client

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=mock_client,
        )

        processor.joint_names = ["_1", "_2", "_3", "_4", "_5", "_6"]
        processor.num_joints = 6
        processor._current_joints = {
            "_1": 0.1, "_2": 0.2, "_3": 0.3, "_4": 0.4, "_5": 0.5, "_6": 0.6
        }

        fake_img = np.asarray(
            Image.open(io.BytesIO(small_valid_jpeg)).convert("RGB"),
            dtype=np.uint8,
        )

        mock_twin = MagicMock()
        processor.cameras = {
            "observation.images.cam_abc123": CameraBinding(
                role="observation.images.cam_abc123",
                twin_uuid="cam-uuid-1",
                twin=mock_twin,
                latest_bytes=small_valid_jpeg,
                latest_image=fake_img.copy(),
                last_ts=time.time(),
            ),
            "observation.images.cam_def456": CameraBinding(
                role="observation.images.cam_def456",
                twin_uuid="cam-uuid-2",
                twin=mock_twin,
                latest_bytes=small_valid_jpeg,
                latest_image=fake_img.copy(),
                last_ts=time.time(),
            ),
        }

        inputs = processor.get_inputs()

        assert "images" in inputs
        assert len(inputs["images"]) == 2
        assert "observation.images.cam_abc123" in inputs["images"]
        assert "observation.images.cam_def456" in inputs["images"]

        mock_twin.get_latest_frame.assert_not_called()

        assert isinstance(inputs["images"]["observation.images.cam_abc123"], np.ndarray)
        assert inputs["images"]["observation.images.cam_abc123"].dtype == np.uint8

        assert "state" in inputs
        assert len(inputs["state"]) == 6

        processor.disconnect()

    def test_get_frames_returns_cached_bytes(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
        small_valid_jpeg: bytes,
    ) -> None:
        """Test that get_frames returns cached bytes when cameras are set up."""
        from cw_processor import CwProcessor, CameraBinding, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        mock_client = MagicMock()
        mock_client.affect.return_value = mock_client

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=mock_client,
        )

        processor.client = mock_client
        processor.robot_twin = MagicMock()

        mock_twin = MagicMock()
        processor.cameras = {
            "cam_front": CameraBinding(
                role="cam_front",
                twin_uuid="cam-uuid-1",
                twin=mock_twin,
                latest_bytes=small_valid_jpeg,
                latest_image=None,
                last_ts=time.time(),
            ),
        }

        frames = processor.get_frames()

        assert "cam_front" in frames
        assert frames["cam_front"] == small_valid_jpeg

        mock_twin.get_latest_frame.assert_not_called()

        processor.disconnect()

    def test_disconnect_stops_camera_threads(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
        small_valid_jpeg: bytes,
    ) -> None:
        """Test that disconnect() properly stops camera threads."""
        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        mock_client = MagicMock()
        mock_camera_twin = MagicMock()
        mock_camera_twin.get_latest_frame.return_value = small_valid_jpeg
        mock_client.twins.get.return_value = mock_camera_twin
        mock_client.affect.return_value = mock_client

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=mock_client,
        )

        processor.client = mock_client
        processor.camera_mapping = {"cam_front": "cam-uuid-1"}
        processor._setup_camera_bindings()

        assert len(processor._camera_threads) == 1
        assert processor._camera_threads[0].is_alive()

        processor.disconnect()

        assert processor._camera_stop.is_set()
        assert len(processor._camera_threads) == 0
        assert len(processor.cameras) == 0
        assert processor.client is None

    def test_camera_binding_initial_joints_in_result(
        self,
        sample_inference_payload: dict[str, Any],
        mock_predict_fn: MagicMock,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test that initial_joints is included in run() result."""
        from cw_processor import CwProcessor, parse_request_payload

        request = parse_request_payload(json.dumps(sample_inference_payload))

        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint=str(temp_checkpoint_dir),
            predict_fn=mock_predict_fn,
            cw=MagicMock(),
        )

        processor.initial_joints = {
            "_1": 0.0, "_2": -1.5, "_3": 1.5, "_4": 0.0, "_5": 0.0, "_6": 1.0
        }

        result = {
            "status": "ok",
            "initial_joints": processor.initial_joints,
        }

        assert "initial_joints" in result
        assert result["initial_joints"]["_2"] == -1.5
