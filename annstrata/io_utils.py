import gzip
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def open_output(path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".gz":
        return gzip.open(output, "wt", encoding="utf-8", newline="")
    return output.open("w", encoding="utf-8", newline="")


def dependency_versions():
    import lightgbm
    import numpy
    import pandas

    versions = {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__, "lightgbm": lightgbm.__version__}
    try:
        import pysam
        versions["pysam"] = pysam.__version__
    except ImportError:
        versions["pysam"] = None
    return versions


def file_record(path, include_sha256=False):
    from .registry import sha256

    value = Path(path).resolve()
    record = {"path": str(value), "bytes": value.stat().st_size}
    if include_sha256:
        record["sha256"] = sha256(value)
    return record


def write_metadata(output, *, command, inputs, model, results, include_input_sha256=False):
    payload = {
        "schema": "annstrata-run-metadata-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "argv": sys.argv,
        "inputs": [file_record(path, include_input_sha256) for path in inputs],
        "model": model,
        "results": results,
        "software": dependency_versions(),
    }
    path = Path(str(output) + ".metadata.json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return path
