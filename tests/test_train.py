"""Tests for train.py operator-config loading (cyberwave.yml config: block)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import train


class TestLoadYmlConfig:
    def _write(self, tmp_path: Path, body: str) -> Path:
        p = tmp_path / "cyberwave.yml"
        p.write_text(textwrap.dedent(body))
        return p

    def test_reads_nested_train_section(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        yml = self._write(
            tmp_path,
            """
            cyberwave-cloud-node:
              config:
                train:
                  max_steps: 5
                  save_freq: 5
                deploy:
                  something: 1
            """,
        )
        assert train._load_yml_config("train", yml_path=yml) == {"max_steps": 5, "save_freq": 5}
        assert train._load_yml_config("deploy", yml_path=yml) == {"something": 1}

    def test_legacy_flat_config_treated_as_train(self, tmp_path: Path) -> None:
        pytest.importorskip("yaml")
        yml = self._write(
            tmp_path,
            """
            cyberwave-cloud-node:
              config:
                max_steps: 3
                log_freq: 1
            """,
        )
        assert train._load_yml_config("train", yml_path=yml) == {"max_steps": 3, "log_freq": 1}
        assert train._load_yml_config("deploy", yml_path=yml) == {}

    def test_missing_file_or_block_returns_empty(self, tmp_path: Path) -> None:
        assert train._load_yml_config("train", yml_path=tmp_path / "nope.yml") == {}
        yml = self._write(tmp_path, "cyberwave-cloud-node:\n  install_script: ./x.sh\n")
        assert train._load_yml_config("train", yml_path=yml) == {}
