import gzip
import hashlib
import json
import os
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = TOOL_ROOT / "models" / "registry.json"


def load_registry():
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def model_root(override=None):
    if override:
        return Path(override).expanduser().resolve()
    if os.environ.get("ANNSTRATA_MODEL_ROOT"):
        return Path(os.environ["ANNSTRATA_MODEL_ROOT"]).expanduser().resolve()
    return (TOOL_ROOT / "models").resolve()


def resolve_artifact(entry, root):
    path = root / entry["path"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Required model artifact was not found: {path}. "
            "Set --model-root or ANNSTRATA_MODEL_ROOT."
        )
    # gzip-compressed weights: "bytes" and "sha256" describe the model content, "compressed_bytes" the file
    expected = entry.get("compressed_bytes") if path.suffix == ".gz" else entry.get("bytes")
    if expected is not None and path.stat().st_size != expected:
        raise RuntimeError(
            f"Model size mismatch for {path}: expected {expected}, got {path.stat().st_size}"
        )
    return path


def _open_content(path):
    """Model content: a .gz file is read decompressed."""
    path = Path(path)
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def load_booster(path):
    """LightGBM booster from a model text file or its gzip-compressed copy."""
    import lightgbm as lgb
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return lgb.Booster(model_str=handle.read())
    return lgb.Booster(model_file=str(path))


def sha256(path, chunk_size=16 * 1024 * 1024, content=False):
    digest = hashlib.sha256()
    with (_open_content(path) if content else Path(path).open("rb")) as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(path, entry):
    expected = entry.get("sha256")
    if not expected:
        return None
    observed = sha256(path, content=True)
    if observed != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected {expected}, got {observed}"
        )
    return observed
