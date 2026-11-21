"""Rebuild the dataset (and outputs) from the split 7z archives in ``archives/``.

The images are too large for GitHub, so they are stored as 20 MB 7z volumes:

    archives/data/data-*.7z.001 ... .055     -> data/raw/{genuine,forged}/...
    archives/outputs/outputs-*.7z.001 ... .003 -> outputs/{split.csv,resnet18_people.pt}

Usage (from the project root):

    pip install py7zr
    python scripts/rebuild_data.py            # skips anything already present
    python scripts/rebuild_data.py --force    # re-extract everything

The notebook calls ``rebuild()`` itself, so running this by hand is optional.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SplitVolumeReader(io.RawIOBase):
    """Read-only, seekable view over ``name.7z.001``, ``.002``, ... as one file."""

    def __init__(self, parts: list[Path]):
        self._files = [open(p, "rb") for p in parts]
        self._sizes = [os.path.getsize(p) for p in parts]
        self._starts = [sum(self._sizes[:i]) for i in range(len(parts))]
        self._size = sum(self._sizes)
        self._pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, buffer):
        view = memoryview(buffer)
        written = 0
        while written < len(view) and self._pos < self._size:
            i = max(j for j, start in enumerate(self._starts) if start <= self._pos)
            f = self._files[i]
            f.seek(self._pos - self._starts[i])
            n = f.readinto(view[written:written + self._sizes[i] - (self._pos - self._starts[i])])
            if not n:
                break
            written += n
            self._pos += n
        return written

    def close(self):
        for f in self._files:
            f.close()
        super().close()


def _parts(folder: Path) -> list[Path]:
    parts = sorted(p for p in folder.glob("*.7z.[0-9][0-9][0-9]"))
    if not parts:
        raise FileNotFoundError(f"No *.7z.001 parts found in {folder}")
    numbers = [int(p.suffix[1:]) for p in parts]
    if numbers != list(range(1, len(parts) + 1)):
        missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
        raise FileNotFoundError(f"Missing parts in {folder}: {missing}")
    return parts


def _extract_7z(folder: Path, dest: Path) -> None:
    import py7zr  # imported lazily so the module can be imported without it

    reader = SplitVolumeReader(_parts(folder))
    try:
        with py7zr.SevenZipFile(reader, mode="r") as archive:
            archive.extractall(path=dest)
    finally:
        reader.close()


def _only_zip(folder: Path) -> Path:
    zips = list(folder.glob("*.zip"))
    if len(zips) != 1:
        raise RuntimeError(f"Expected one zip in {folder}, found {zips}")
    return zips[0]


def rebuild_data(root: Path = PROJECT_ROOT, force: bool = False) -> Path:
    """Ensure ``data/manifest.csv`` and ``data/raw/`` exist. Returns the data dir."""
    data_dir = root / "data"
    if (data_dir / "raw").is_dir() and not force:
        print(f"data: already present in {data_dir / 'raw'}")
        return data_dir

    with tempfile.TemporaryDirectory(dir=root) as tmp:
        tmp = Path(tmp)
        print("data: extracting 7z volumes (1.15 GB, takes a few minutes)...")
        _extract_7z(root / "archives" / "data", tmp)
        # 7z -> data-*.zip (Drive export) -> data/{manifest.csv, signature-rec-colab.zip}
        with zipfile.ZipFile(_only_zip(tmp)) as outer:
            outer.extractall(tmp)
        (tmp / _only_zip(tmp).name).unlink()
        inner = tmp / "data" / "signature-rec-colab.zip"
        print("data: unzipping images into data/raw ...")
        if force and (data_dir / "raw").exists():
            shutil.rmtree(data_dir / "raw")
        data_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(inner) as images:
            images.extractall(data_dir)  # writes raw/... and manifest.csv
    print(f"data: done -> {data_dir}")
    return data_dir


def rebuild_outputs(root: Path = PROJECT_ROOT, force: bool = False) -> Path:
    """Ensure ``outputs/split.csv`` and ``outputs/resnet18_people.pt`` exist."""
    out_dir = root / "outputs"
    wanted = [out_dir / "split.csv", out_dir / "resnet18_people.pt"]
    if all(p.exists() for p in wanted) and not force:
        print(f"outputs: already present in {out_dir}")
        return out_dir

    with tempfile.TemporaryDirectory(dir=root) as tmp:
        tmp = Path(tmp)
        _extract_7z(root / "archives" / "outputs", tmp)
        with zipfile.ZipFile(_only_zip(tmp)) as z:
            z.extractall(tmp)
        out_dir.mkdir(exist_ok=True)
        for f in (tmp / "outputs").iterdir():
            shutil.copy2(f, out_dir / f.name)
    print(f"outputs: done -> {out_dir}")
    return out_dir


def rebuild(root: Path = PROJECT_ROOT, force: bool = False) -> None:
    rebuild_outputs(root, force)
    rebuild_data(root, force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-extract even if files exist")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="project root")
    args = parser.parse_args()
    rebuild(args.root, args.force)
