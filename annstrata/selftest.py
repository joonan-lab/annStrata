import json
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import resolve_artifact, verify_artifact


TESTS = Path(__file__).resolve().parents[1] / "tests"
EXPECTED_PATH = TESTS / "expected.json"
ROLES = ("model", "quantile_lo", "quantile_med", "quantile_hi", "calibration", "spec")


def _test_str(registry, root, expected):
    from .str_score import load_bundle, read_vcf, score_parsed
    bundle, _ = load_bundle(registry, root)
    header, samples, parsed = read_vcf(str(TESTS / "str_probe.vcf"))
    out_fields, rows, summary = score_parsed(parsed, samples, bundle, 4)
    observed = [";".join(f) for f in (of[0] for of in out_fields)]
    if observed != expected["str"]["fields"]:
        raise AssertionError(f"STR probe fields differ:\n observed {observed}\n expected {expected['str']['fields']}")
    print(f"  PASS STR probe: {len(parsed)} records ({summary['scored_diploid']} diploid, {summary['scored_haploid']} haploid, {summary['out_of_scope_chrY_chrM']} out-of-scope chrY/chrM) reproduce frozen fields")


def _test_sv(registry, root, expected):
    """Scores the frozen feature table of the SV probe (feature_builder output for tests/sv_probe.vcf) and compares ASEP values."""
    from .sv_score import load_bundle, score_frame
    bundle, _ = load_bundle(registry, root)
    frame = pd.read_csv(TESTS / "sv_probe.features.tsv", sep="\t")
    p, reasons, _, _, _, summary = score_frame(frame, bundle, 4)
    observed = [None if not (x == x) else round(float(x), 4) for x in p]
    want = expected["sv"]["error_probability"]
    if len(observed) != len(want) or any((a is None) != (b is None) or (a is not None and abs(a - b) > 1e-4) for a, b in zip(observed, want)):
        raise AssertionError(f"SV probe error probabilities differ:\n observed {observed}\n expected {want}")
    print(f"  PASS SV probe: {summary['records']} records ({summary['scored']} scored) reproduce frozen ASEP values")


def _check_specs(registry, root, component, checksum):
    names = []
    for name, entry in registry["models"].items():
        if component != "all" and name != component:
            continue
        for role in ROLES:
            if role not in entry:
                continue
            path = resolve_artifact(entry[role], root)
            if checksum:
                verify_artifact(path, entry[role])
            names.append(str(path))
    print(f"  PASS artifacts: {len(names)} files (checksum={'on' if checksum else 'off'})")


def run_selftest(registry, root, component="all", checksum=False):
    with EXPECTED_PATH.open(encoding="utf-8") as handle:
        expected = json.load(handle)
    print("annStrata self-test")
    _check_specs(registry, root, component, checksum)
    if component in ("all", "str"):
        _test_str(registry, root, expected)
    if component in ("all", "sv"):
        _test_sv(registry, root, expected)
    print("SELF-TEST PASSED")
