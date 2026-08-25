"""Tests for cw_processor module - inference request parsing and utilities."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cw_processor import (
    InferenceRequest,
    _coerce_list,
    _coerce_mapping,
    _extract_joint_positions,
    _get_param,
    _normalize_sampling_calibration,
    load_json_argument,
    normalize_twin_calibration,
    parse_request_payload,
)


class TestLoadJsonArgument:
    """Tests for load_json_argument function."""

    def test_load_from_json_string(self) -> None:
        """Test loading JSON from a string."""
        payload = '{"robot_twin_uuid": "test-uuid", "instruction": "test"}'
        result = load_json_argument(payload)
        assert result == payload

    def test_load_from_file(self, tmp_path: Path) -> None:
        """Test loading JSON from a file path."""
        json_content = '{"robot_twin_uuid": "file-uuid", "instruction": "from file"}'
        json_file = tmp_path / "params.json"
        json_file.write_text(json_content)

        result = load_json_argument(str(json_file))
        assert result == json_content

    def test_load_nonexistent_file_returns_as_string(self) -> None:
        """Non-existent path is returned as-is (for JSON string handling)."""
        result = load_json_argument("/nonexistent/path.json")
        assert result == "/nonexistent/path.json"


class TestParseRequestPayload:
    """Tests for parse_request_payload function."""

    def test_parse_snake_case_payload(
        self, sample_inference_payload: dict[str, Any]
    ) -> None:
        """Test parsing a payload with snake_case keys."""
        raw = json.dumps(sample_inference_payload)
        request = parse_request_payload(raw)

        assert isinstance(request, InferenceRequest)
        assert request.robot_twin_uuid == "b10e8ffa-f58c-49e0-a9c0-f76ffbed0356"
        assert request.instruction == "pick up the red block"
        assert request.max_steps == 100
        assert request.mode == "live"
        assert request.calibration_robot_type == "follower"
        assert request.action_sleep_seconds == 0.1
        assert request.actions_per_cycle == 25
        assert request.inference_loop is True

    def test_parse_camel_case_payload(
        self, sample_inference_payload_camel_case: dict[str, Any]
    ) -> None:
        """Test parsing a payload with camelCase keys."""
        raw = json.dumps(sample_inference_payload_camel_case)
        request = parse_request_payload(raw)

        assert request.robot_twin_uuid == "b10e8ffa-f58c-49e0-a9c0-f76ffbed0356"
        assert request.max_steps == 50
        assert request.wait_for_joint_update_seconds == 2.0
        assert request.action_sleep_seconds == 0.2
        assert request.actions_per_cycle == 10
        assert request.inference_loop is False

    def test_parse_minimal_payload(self) -> None:
        """Test parsing a minimal payload with only required fields."""
        raw = json.dumps({"robot_twin_uuid": "minimal-uuid"})
        request = parse_request_payload(raw)

        assert request.robot_twin_uuid == "minimal-uuid"
        assert request.instruction == "perform the requested task"  # default
        assert request.max_steps == 1  # default
        assert request.action_sleep_seconds == 0.1  # default

    def test_parse_twin_uuid_fallback(self) -> None:
        """Test that twin_uuid is accepted as fallback for robot_twin_uuid."""
        raw = json.dumps({"twin_uuid": "fallback-uuid"})
        request = parse_request_payload(raw)
        assert request.robot_twin_uuid == "fallback-uuid"

    def test_parse_missing_robot_uuid_raises(self) -> None:
        """Test that missing robot UUID raises ValueError."""
        raw = json.dumps({"instruction": "no robot"})
        with pytest.raises(ValueError, match="robot_twin_uuid"):
            parse_request_payload(raw)

    def test_parse_camera_endpoints_by_role(self) -> None:
        """Test parsing camera endpoints by role."""
        raw = json.dumps(
            {
                "robot_twin_uuid": "test",
                "camera_endpoints_by_role": {
                    "primary": "cam-1",
                    "wrist": "cam-2",
                },
            }
        )
        request = parse_request_payload(raw)

        assert request.camera_endpoints_by_role == {
            "primary": "cam-1",
            "wrist": "cam-2",
        }
        # camera_twin_uuids should be populated from endpoints
        assert "cam-1" in request.camera_twin_uuids
        assert "cam-2" in request.camera_twin_uuids

    def test_parse_camera_slots_preserves_order_and_deduplicates(self) -> None:
        request = parse_request_payload(
            json.dumps(
                {
                    "robot_twin_uuid": "robot-1",
                    "camera_slots": ["primary_camera", "wrist_camera", "primary_camera"],
                }
            )
        )

        assert request.camera_slots == ["primary_camera", "wrist_camera"]

    def test_parse_camera_endpoints_extracts_uuid_from_url(self) -> None:
        """Test that UUID is extracted from latest-frame URLs."""
        raw = json.dumps(
            {
                "robot_twin_uuid": "test",
                "camera_endpoints_by_role": {
                    "primary": "/twins/abc-123/latest-frame",
                },
            }
        )
        request = parse_request_payload(raw)
        assert "abc-123" in request.camera_twin_uuids

    def test_parse_twin_calibration(
        self, sample_inference_payload: dict[str, Any]
    ) -> None:
        """Test parsing twin calibration data."""
        raw = json.dumps(sample_inference_payload)
        request = parse_request_payload(raw)

        assert "_1" in request.twin_calibration
        assert request.twin_calibration["_1"]["min"] == -3.14
        assert request.twin_calibration["_1"]["max"] == 3.14


class TestHelperFunctions:
    """Tests for helper functions in cw_processor."""

    def test_get_param_snake_case(self) -> None:
        """Test _get_param with snake_case key."""
        params = {"robot_twin_uuid": "snake"}
        result = _get_param(params, "robot_twin_uuid", "robotTwinUuid")
        assert result == "snake"

    def test_get_param_camel_case(self) -> None:
        """Test _get_param with camelCase key."""
        params = {"robotTwinUuid": "camel"}
        result = _get_param(params, "robot_twin_uuid", "robotTwinUuid")
        assert result == "camel"

    def test_get_param_default(self) -> None:
        """Test _get_param returns default when key not found."""
        params = {}
        result = _get_param(params, "missing", "alsoMissing", "default_value")
        assert result == "default_value"

    def test_coerce_list_from_none(self) -> None:
        """Test _coerce_list with None."""
        assert _coerce_list(None) == []

    def test_coerce_list_from_string(self) -> None:
        """Test _coerce_list with string."""
        assert _coerce_list("single") == ["single"]

    def test_coerce_list_from_list(self) -> None:
        """Test _coerce_list with list."""
        assert _coerce_list(["a", "b", "c"]) == ["a", "b", "c"]

    def test_coerce_list_filters_none(self) -> None:
        """Test _coerce_list filters None values."""
        assert _coerce_list(["a", None, "b"]) == ["a", "b"]

    def test_coerce_mapping_from_none(self) -> None:
        """Test _coerce_mapping with None."""
        assert _coerce_mapping(None) == {}

    def test_coerce_mapping_from_dict(self) -> None:
        """Test _coerce_mapping with dict."""
        assert _coerce_mapping({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_coerce_mapping_raises_for_non_dict(self) -> None:
        """Test _coerce_mapping raises for non-dict."""
        with pytest.raises(ValueError, match="Expected object"):
            _coerce_mapping("not a dict")


class TestNormalizeTwinCalibration:
    """Tests for normalize_twin_calibration function."""

    def test_normalize_simple_calibration(self) -> None:
        """Test normalizing simple calibration dict."""
        calibration = {
            "joint_1": {"min": -1.0, "max": 1.0},
            "joint_2": {"min": -2.0, "max": 2.0},
        }
        result = normalize_twin_calibration(calibration)

        assert result["joint_1"]["min"] == -1.0
        assert result["joint_1"]["max"] == 1.0

    def test_normalize_calibration_with_lower_upper(self) -> None:
        """Test normalizing calibration with lower/upper keys."""
        calibration = {
            "joint_1": {"lower": -1.0, "upper": 1.0},
        }
        result = _normalize_sampling_calibration(calibration)

        assert result["joint_1"]["min"] == -1.0
        assert result["joint_1"]["max"] == 1.0

    def test_normalize_leader_follower_calibration(self) -> None:
        """Test normalizing calibration with leader/follower structure."""
        calibration = {
            "leader": {"joint_1": {"min": -1.0, "max": 1.0}},
            "follower": {"joint_1": {"min": -2.0, "max": 2.0}},
        }
        result = normalize_twin_calibration(calibration, robot_type="follower")

        assert result["joint_1"]["min"] == -2.0
        assert result["joint_1"]["max"] == 2.0

    def test_normalize_none_calibration(self) -> None:
        """Test normalizing None calibration."""
        result = normalize_twin_calibration(None)
        assert result == {}


class TestExtractJointPositions:
    """Tests for _extract_joint_positions function."""

    def test_extract_from_positions_dict(
        self, sample_mqtt_joint_payload: dict[str, Any]
    ) -> None:
        """Test extracting from standard MQTT positions format."""
        result = _extract_joint_positions(sample_mqtt_joint_payload)

        assert result["_1"] == pytest.approx(0.01)
        assert result["_2"] == pytest.approx(-1.54)
        assert result["_6"] == pytest.approx(0.95)

    def test_extract_from_joint_states_dict(self) -> None:
        """Test extracting from legacy joint_states format."""
        payload = {
            "joint_states": {
                "joint_1": {"position": 0.5},
                "joint_2": {"position": -0.5},
            }
        }
        result = _extract_joint_positions(payload)

        assert result["joint_1"] == 0.5
        assert result["joint_2"] == -0.5

    def test_extract_from_single_joint(self) -> None:
        """Test extracting from single joint format."""
        payload = {
            "joint_name": "gripper",
            "joint_state": {"position": 0.8},
        }
        result = _extract_joint_positions(payload)
        assert result["gripper"] == 0.8

    def test_extract_empty_payload(self) -> None:
        """Test extracting from empty payload."""
        result = _extract_joint_positions({})
        assert result == {}
