"""SV scoring with the annStrata SV model (LightGBM).

Input: a single-sample short-read SV VCF (Manta; Delly and LUMPY VCFs are parsed to the same feature concepts) plus the GRCh38
reference and the GIAB / RepeatMasker BEDs (resources/README.md), or a precomputed feature table written by feature_builder.py.
Outputs (either or both): the input VCF with one FORMAT field appended (ASEP; every record kept), and/or a per-call TSV.
Scope: DEL / INS / DUP calls of 50 bp - 5 Mb (the size window of the training labels) on autosomes and chrX (Manta has no sex
input, so neither does the model); other records are kept in the VCF with ASEP "." and are
absent from the TSV.
Per call: ASEP / TSV error_probability = calibrated P(call is discordant with the 2-of-3 long-read consensus) from a LightGBM binary
classifier + global isotonic map (fitted on sample-split out-of-fold predictions).
"""
import gzip
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import lightgbm as lgb  # noqa: F401  (fails early when LightGBM is missing; boosters come from registry.load_booster)
import numpy as np
import pandas as pd

from .io_utils import write_metadata
from .registry import load_booster, resolve_artifact

SENTINEL = -99999.0
SCORED_CONTIGS = {f"chr{i}" for i in list(range(1, 23)) + ["X"]} | {str(i) for i in list(range(1, 23)) + ["X"]}
SUPPORTED = ("DEL", "INS", "DUP")
SIZE_MIN, SIZE_MAX = 50, 5_000_000
SV_ROLES = ("model", "calibration", "spec")
TSV_COLUMNS = ["sample", "chrom", "pos", "end", "svtype", "svlen", "filter", "error_probability"]
VCF_HEADER = [
    '##FORMAT=<ID=ASEP,Number=1,Type=Float,Description="annStrata SV error probability: P(call is discordant with a 2-of-3 long-read consensus of Sniffles2, cuteSV and Sawfish)">',
    '##annStrataNote="ASEP is given for DEL/INS/DUP calls of 50 bp - 5 Mb on autosomes and chrX; other records (INV, BND, chrY, chrM, other contigs, sizes outside the window) carry ."',
]


def _opener(path, mode):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def load_bundle(registry, root):
    """LightGBM booster + calibration knots + spec. Returns (bundle dict, {role: path})."""
    entry = registry["models"]["sv"]
    paths = {role: resolve_artifact(entry[role], root) for role in SV_ROLES}
    booster = load_booster(paths["model"])
    with paths["spec"].open(encoding="utf-8") as handle:
        spec = json.load(handle)
    if booster.feature_name() != spec["features"]:
        raise RuntimeError("SV model: feature order does not match spec.json")
    with paths["calibration"].open(encoding="utf-8") as handle:
        cal = json.load(handle)
    return {"booster": booster, "cal_x": np.asarray(cal["x"], dtype=np.float64), "cal_y": np.asarray(cal["y"], dtype=np.float64), "spec": spec}, paths


def build_features(args, output):
    builder = Path(__file__).with_name("feature_builder.py")
    command = [sys.executable, str(builder), "--vcf", args.input, "--caller", args.caller, "--fasta", args.fasta, "--giab-dir", args.giab_dir,
               "--repeatmasker", args.repeatmasker, "--neighbor-radius", str(args.neighbor_radius), "--win", str(args.window), "-o", str(output)]
    if args.sample:
        command.extend(["--sample", args.sample])
    subprocess.run(command, check=True)


def feature_matrix(frame, spec):
    """32-feature float32 matrix in spec order (sentinel -> NaN); the spec maps feature names to feature_builder columns."""
    source = spec.get("source_column", {})
    cols = [source.get(f, f) for f in spec["features"]]
    absent = [c for c in cols if c not in frame.columns]
    if absent:
        raise RuntimeError(f"Feature table is missing {len(absent)} required columns: {', '.join(absent)}")
    X = frame[cols].to_numpy(dtype=np.float64)
    X[X == SENTINEL] = np.nan
    return X.astype(np.float32)


def scope_mask(frame):
    """Rows the model scores: DEL/INS/DUP on autosomes and chrX with 50 bp <= |SVLEN| <= 5 Mb (SVLEN from svlen_log = log10(|SVLEN| + 1))."""
    parts = frame["variant_ID"].str.split(":", expand=True)
    chrom, svtype = parts[0], parts[2]
    svlen = np.where(frame["svlen_log"].to_numpy() == SENTINEL, np.nan, np.rint(10.0 ** frame["svlen_log"].to_numpy(dtype=np.float64) - 1.0))
    reasons = pd.Series("scored", index=frame.index)
    reasons[~svtype.isin(SUPPORTED)] = "unsupported_svtype"
    reasons[~chrom.isin(SCORED_CONTIGS)] = "out_of_scope_contig"
    reasons[(reasons == "scored") & (np.isnan(svlen) | (svlen < SIZE_MIN) | (svlen > SIZE_MAX))] = "size_outside_50bp_5Mb"
    return reasons, chrom, svtype, svlen


def score_frame(frame, bundle, threads):
    """Calibrated error probability per row (NaN outside scope), the scope reason per row, and a summary."""
    reasons, chrom, svtype, svlen = scope_mask(frame)
    use = (reasons == "scored").to_numpy()
    p = np.full(len(frame), np.nan)
    if use.any():
        X = feature_matrix(frame.loc[use], bundle["spec"])
        p[use] = np.interp(bundle["booster"].predict(X, num_threads=threads), bundle["cal_x"], bundle["cal_y"])
    summary = {"records": int(len(frame)), "scored": int(use.sum())}
    for reason, n in reasons.value_counts().items():
        if reason != "scored":
            summary[reason] = int(n)
    return p, reasons, chrom, svtype, svlen, summary


def write_vcf(path, vcf_in, frame, p, sample, model_id, summary):
    """Append ASEP to the input VCF; rows are matched to records by the builder's vcf_index (0-based index over all input records)."""
    by_index = dict(zip(frame["vcf_index"].astype(int).tolist(), p.tolist()))
    header, out_lines = [], []
    with _opener(vcf_in, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                header.append(line.rstrip("\n"))
            else:
                out_lines.append(line.rstrip("\n"))
    samples = header[-1].split("\t")[9:]
    if sample not in samples:
        raise RuntimeError(f"sample {sample} not in VCF columns {samples}")
    si = samples.index(sample)
    new_header = VCF_HEADER + [f'##annStrataCommandLine=<ID=annStrata,Version=1.0.0-{model_id},Date="{time.strftime("%Y-%m-%d")}",Summary="{json.dumps(summary).replace(chr(34), "")}",CommandLine="{" ".join(sys.argv)}">']
    format_lines = [i for i, h in enumerate(header) if h.startswith("##FORMAT")]
    cut = (format_lines[-1] + 1) if format_lines else len(header) - 1
    header2 = header[:cut] + new_header + header[cut:]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _opener(path, "wt") as out:
        out.write("\n".join(header2) + "\n")
        for i, line in enumerate(out_lines):
            fields = line.split("\t")
            v = by_index.get(i, np.nan)
            values = [f"{v:.3f}" if (j == si and v == v) else "." for j in range(len(samples))]
            row = fields[:8] + [fields[8] + ":ASEP"] + [fields[9 + j] + ":" + values[j] for j in range(len(samples))]
            out.write("\t".join(row) + "\n")


def write_tsv(path, frame, p, chrom, svtype, svlen, reasons):
    use = (reasons == "scored").to_numpy()
    parts = frame["variant_ID"].str.split(":", expand=True)
    table = pd.DataFrame({"sample": frame["sample"], "chrom": chrom, "pos": parts[1].astype(int), "end": parts[3].where(parts[3] != "None", "."), "svtype": svtype,
                          "svlen": np.where(np.isnan(svlen), ".", svlen.astype(object)), "filter": frame["filter"] if "filter" in frame.columns else ".",
                          "error_probability": [f"{x:.4f}" for x in p]})[use]
    table["svlen"] = table["svlen"].map(lambda x: x if x == "." else str(int(x)))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _opener(path, "wt") as out:
        table.to_csv(out, sep="\t", index=False, columns=TSV_COLUMNS)


def score_sv(args, registry, root):
    t0 = time.time()
    if not (args.out_vcf or args.out_tsv):
        raise ValueError("Specify --out-vcf and/or --out-tsv")
    bundle, paths = load_bundle(registry, root)
    model_id = registry["models"]["sv"]["model"]["sha256"][:12]
    is_table = args.input.endswith((".tsv", ".tsv.gz"))
    if is_table:
        if args.out_vcf:
            raise ValueError("--out-vcf needs a VCF input; a feature table can only be written as TSV")
        frame = pd.read_csv(args.input, sep="\t")
        inputs = [args.input]
    else:
        tool_root = Path(__file__).resolve().parents[1]
        args.giab_dir = args.giab_dir or str(tool_root / "resources/giab_prepared")
        args.repeatmasker = args.repeatmasker or str(tool_root / "resources/repeatmasker/rmsk_class.bed.gz")
        if not (args.caller and args.fasta):
            raise RuntimeError("VCF input needs --caller and --fasta (GRCh38 FASTA with .fai; see resources/README.md)")
        with tempfile.TemporaryDirectory(prefix="annstrata-sv-") as temp_dir:
            feature_path = Path(args.features_out) if args.features_out else Path(temp_dir) / "features.tsv"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            build_features(args, feature_path)
            frame = pd.read_csv(feature_path, sep="\t")
        inputs = [args.input] + ([args.features_out] if args.features_out else [])
    if frame.empty:
        raise RuntimeError(f"No SV records were parsed from {args.input}")
    p, reasons, chrom, svtype, svlen, summary = score_frame(frame, bundle, args.threads)
    sample = str(frame["sample"].iloc[0])
    if args.out_vcf:
        write_vcf(args.out_vcf, args.input, frame, p, sample, model_id, summary)
    if args.out_tsv:
        write_tsv(args.out_tsv, frame, p, chrom, svtype, svlen, reasons)
    summary["minutes"] = round((time.time() - t0) / 60, 2)
    spec = bundle["spec"]
    model_meta = {
        "family": "SV", "model_version": f"1.0.0-{model_id}",
        "endpoints": {"error_probability": "ASEP and TSV error_probability = calibrated P(call is discordant with the 2-of-3 long-read consensus); higher is worse"},
        "truth": spec.get("truth"), "training": spec.get("training_rows"), "scope": spec.get("scope"),
        "artifacts": [{"name": "sv", "role": r, "path": str(pth), "bytes": pth.stat().st_size, "sha256": registry["models"]["sv"][r].get("sha256")} for r, pth in paths.items()],
    }
    outputs = [x for x in (args.out_vcf, args.out_tsv) if x]
    metadata = write_metadata(outputs[0], command="sv", inputs=inputs, model=model_meta, results=summary, include_input_sha256=args.hash_input)
    return summary, metadata
