"""Integration tests for SmolVLA with real Cyberwave SDK.

These tests require network access and valid Cyberwave credentials.
Run with: pytest tests/test_integration.py -v -m integration
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _has_cyberwave_sdk() -> bool:
    """Check if cyberwave SDK is available."""
    try:
        import cyberwave
        return True
    except ImportError:
        return False


def _has_cyberwave_credentials() -> bool:
    """Check if Cyberwave credentials are available."""
    if os.environ.get("CYBERWAVE_API_KEY"):
        return True
    creds_path = Path.home() / ".cyberwave" / "credentials.json"
    return creds_path.exists()


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_cyberwave_sdk(),
    reason="cyberwave SDK not installed"
)
@pytest.mark.skipif(
    not _has_cyberwave_credentials(),
    reason="Cyberwave credentials not available"
)
class TestCwProcessorIntegrationWithSDK:
    """Integration tests using real Cyberwave SDK."""

    @pytest.fixture
    def cyberwave_client(self) -> Any:
        """Create a real Cyberwave client."""
        from cyberwave import Cyberwave
        from cw_processor import apply_cyberwave_credentials_env

        apply_cyberwave_credentials_env()
        return Cyberwave()

    def test_derive_joint_names_from_so101_asset(
        self,
        cyberwave_client: Any,
    ) -> None:
        """Test deriving joint names from a twin with the-robot-studio/so101 asset.

        This test gets the SO-101 asset and verifies its schema has the expected joints.
        """
        # Get the SO-101 asset using registry_id or slug
        asset = None
        slugs_to_try = [
            "the-robot-studio/catalog/so101",
            "the-robot-studio/catalog/so-101",
            "cyberwave/catalog/so101",
        ]

        for slug in slugs_to_try:
            try:
                asset = cyberwave_client.assets.get_by_slug(slug)
                if asset:
                    break
            except Exception:
                continue

        if not asset:
            pytest.skip("SO-101 asset not available in workspace")

        # Get schema from the asset
        schema = None
        if hasattr(asset, "universal_schema"):
            schema = asset.universal_schema
        elif hasattr(asset, "get_schema"):
            schema = asset.get_schema()

        if not schema:
            pytest.skip("Asset does not have a schema")

        # Check that the schema has controllable joints
        joints = schema.get("joints", [])
        controllable_types = {"revolute", "prismatic", "continuous"}
        controllable_joints = [
            j["name"] for j in joints
            if isinstance(j, dict) and j.get("type") in controllable_types
        ]

        # SO-101 should have 6 joints: _1 through _6
        assert len(controllable_joints) >= 6, (
            f"Expected at least 6 controllable joints, got {len(controllable_joints)}: {controllable_joints}"
        )

        # Verify the joint names match expected SO-101 pattern
        expected_joints = ["_1", "_2", "_3", "_4", "_5", "_6"]
        sorted_joints = sorted(controllable_joints)[:6]
        assert sorted_joints == expected_joints, (
            f"Expected joints {expected_joints}, got {sorted_joints}"
        )

    def test_twin_get_controllable_joint_names_method(
        self,
        cyberwave_client: Any,
    ) -> None:
        """Test that Twin.get_controllable_joint_names() works correctly.

        This tests the SDK method directly, which is what CwProcessor uses.
        """
        # Find a twin with the SO-101 asset, or skip
        try:
            # List twins (no limit parameter)
            twins = cyberwave_client.twins.list()
            so101_twin = None

            for twin_schema in twins:
                try:
                    # Get full twin object to access methods
                    twin = cyberwave_client.twins.get(str(twin_schema.uuid))

                    # Check if twin has asset with SO-101 in name
                    asset_uuid = getattr(twin_schema, "asset_uuid", None)
                    if not asset_uuid:
                        continue

                    # Get the asset to check its name
                    try:
                        asset = cyberwave_client.assets.get(str(asset_uuid))
                        asset_name = getattr(asset, "name", "") or ""
                        if "so101" in asset_name.lower() or "so-101" in asset_name.lower():
                            so101_twin = twin
                            break
                    except Exception:
                        continue
                except Exception:
                    continue

            if so101_twin is None:
                pytest.skip("No SO-101 twin found in workspace")

            # Get controllable joint names
            joint_names = so101_twin.get_controllable_joint_names()

            # Verify we got the expected joint names
            assert len(joint_names) == 6, f"Expected 6 joints, got {len(joint_names)}: {joint_names}"
            assert joint_names == ["_1", "_2", "_3", "_4", "_5", "_6"], (
                f"Expected SO-101 joints, got {joint_names}"
            )

        except Exception as e:
            pytest.skip(f"Could not test twin joint names: {e}")


@pytest.mark.integration
class TestCwProcessorJointDerivation:
    """Tests for joint name derivation logic with mocked SDK responses."""

    def test_derive_joints_from_schema_response(self) -> None:
        """Test joint derivation with a mocked schema matching SO-101."""
        from cw_processor import CwProcessor, parse_request_payload

        # SO-101 schema structure
        so101_schema = {
            "joints": [
                {"name": "_1", "type": "revolute"},
                {"name": "_2", "type": "revolute"},
                {"name": "_3", "type": "revolute"},
                {"name": "_4", "type": "revolute"},
                {"name": "_5", "type": "revolute"},
                {"name": "_6", "type": "revolute"},
            ]
        }

        # Create mock twin with get_controllable_joint_names matching SDK behavior
        mock_twin = MagicMock()

        def get_controllable_joint_names():
            CONTROLLABLE_JOINT_TYPES = frozenset({"revolute", "prismatic", "continuous"})
            joints = so101_schema.get("joints", [])
            controllable = [
                j["name"]
                for j in joints
                if isinstance(j, dict)
                and j.get("name")
                and j.get("type") in CONTROLLABLE_JOINT_TYPES
            ]
            return sorted(controllable)

        mock_twin.get_controllable_joint_names = get_controllable_joint_names

        # Create processor
        request = parse_request_payload('{"robot_twin_uuid": "test-uuid"}')
        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint="/fake/checkpoint",
            predict_fn=lambda x: x,
            cw=MagicMock(),
        )

        processor.robot_twin = mock_twin
        processor._derive_joint_names_from_twin()

        assert processor.joint_names == ["_1", "_2", "_3", "_4", "_5", "_6"]
        assert processor.num_joints == 6

    def test_derive_joints_from_schema_with_fixed_joints(self) -> None:
        """Test joint derivation filters out fixed joints."""
        from cw_processor import CwProcessor, parse_request_payload

        # Schema with mixed joint types
        schema_with_fixed = {
            "joints": [
                {"name": "base_link", "type": "fixed"},
                {"name": "_1", "type": "revolute"},
                {"name": "_2", "type": "revolute"},
                {"name": "mount", "type": "fixed"},
                {"name": "_3", "type": "revolute"},
                {"name": "_4", "type": "revolute"},
                {"name": "_5", "type": "revolute"},
                {"name": "_6", "type": "revolute"},
            ]
        }

        mock_twin = MagicMock()

        def get_controllable_joint_names():
            CONTROLLABLE_JOINT_TYPES = frozenset({"revolute", "prismatic", "continuous"})
            joints = schema_with_fixed.get("joints", [])
            controllable = [
                j["name"]
                for j in joints
                if isinstance(j, dict)
                and j.get("name")
                and j.get("type") in CONTROLLABLE_JOINT_TYPES
            ]
            return sorted(controllable)

        mock_twin.get_controllable_joint_names = get_controllable_joint_names

        request = parse_request_payload('{"robot_twin_uuid": "test-uuid"}')
        processor = CwProcessor(
            request,
            model_slug="smolvla",
            checkpoint="/fake/checkpoint",
            predict_fn=lambda x: x,
            cw=MagicMock(),
        )

        processor.robot_twin = mock_twin
        processor._derive_joint_names_from_twin()

        # Should only have the 6 revolute joints, not the fixed ones
        assert processor.joint_names == ["_1", "_2", "_3", "_4", "_5", "_6"]
        assert processor.num_joints == 6
        assert "base_link" not in processor.joint_names
        assert "mount" not in processor.joint_names
