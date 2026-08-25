"""Throwaway repro: does tarfile.add() follow the `checkpoints/last` symlink?"""

import os
import tarfile
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as td:
    base = Path(td) / "checkpoints"
    (base / "050000" / "pretrained_model").mkdir(parents=True)
    (base / "050000" / "pretrained_model" / "adapter_model.safetensors").write_bytes(
        os.urandom(2_900_000)
    )
    (base / "050000" / "pretrained_model" / "config.json").write_text('{"a": 1}')
    os.symlink("050000", base / "last")

    ckpt = base / "last"
    print("Path.exists():", ckpt.exists())
    print("is_symlink():", ckpt.is_symlink(), " is_dir():", ckpt.is_dir())

    out = Path(td) / "out.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        tar.add(ckpt, arcname="checkpoint")

    n = out.stat().st_size
    print(f"artifact: {n} bytes -> {n / (1024 * 1024):.1f}MB")
    with tarfile.open(out) as t:
        for m in t.getmembers():
            print(f"  member={m.name!r} type={m.type!r} link={m.linkname!r} size={m.size}")

    # Control: what it looks like when the real dir is tarred
    out2 = Path(td) / "out2.tar.gz"
    with tarfile.open(out2, "w:gz") as tar:
        tar.add(base / "050000", arcname="checkpoint")
    n2 = out2.stat().st_size
    print(f"control (real dir): {n2} bytes -> {n2 / (1024 * 1024):.1f}MB")
