"""STR scoring with the annStrata direct model (LightGBM).

Input: ExpansionHunter / DRAGEN Repeat Genotyper VCF (single- or multi-sample) or a parsed ExpansionHunter caller table
(TSV with columns chrom pos end filter repid ru rl ref repcn repci adsp adfl adir lc so).
Outputs (either or both): the input VCF with four FORMAT fields appended (ASES ASEL ASLS ASLL; every record kept), and/or a
per-allele TSV. Scope: autosomes and chrX; chrY/chrM records are kept but not scored. A haploid call (one REPCN value, e.g. chrX of a male
sample genotyped with its sex) is scored as a homozygous call with the same values in both allele slots, as in training.
Per allele: ASES/ASEL = calibrated P(|EH length - long-read length| > max(1 RU, 0.5 x long-read length)) from a LightGBM binary
classifier + global isotonic map; ASLS/ASLL = predicted long-read length (bp) median,lo80,hi80 from three LightGBM quantile
regressors of r = log2(max(T,1)/max(EH,1)). shorter/longer follow REPCN ascending. The tolerance (alpha 0.5, 1 RU floor) is fixed at training.
"""
import gzip
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb  # noqa: F401  (fails early when LightGBM is missing; boosters come from registry.load_booster)
import numpy as np

from . import __version__
from . import str_features as F
from .io_utils import write_metadata
from .registry import load_booster, resolve_artifact

ALPHA, FLOOR_RU = 0.5, 1
VCF_FORMAT_HEADER = [
    f'##FORMAT=<ID=ASES,Number=1,Type=Float,Description="annStrata error probability, shorter allele: P(|EH length - long-read length| > max({FLOOR_RU} RU, {ALPHA} x long-read length))">',
    f'##FORMAT=<ID=ASEL,Number=1,Type=Float,Description="annStrata error probability, longer allele: P(|EH length - long-read length| > max({FLOOR_RU} RU, {ALPHA} x long-read length))">',
    '##FORMAT=<ID=ASLS,Number=3,Type=Integer,Description="annStrata predicted long-read length (bp), shorter allele: median,lo80,hi80">',
    '##FORMAT=<ID=ASLL,Number=3,Type=Integer,Description="annStrata predicted long-read length (bp), longer allele: median,lo80,hi80">',
    '##annStrataNote="shorter/longer follow REPCN ascending order (ties keep VCF order), not VCF allele order; a haploid call (one REPCN value, e.g. male chrX) is scored as a homozygous call; chrY/chrM are outside the model scope and carry . in all four fields, as do unscorable calls; existing fields are unchanged"',
]
OUT_OF_SCOPE = {"chrY", "chrM", "chrMT", "Y", "M", "MT"}   # chrX is scored; the catalog has no chrY/chrM loci
TSV_COLUMNS = ["sample", "chrom", "pos", "repid", "ru", "allele", "so", "eh_repcn", "eh_bp", "error_probability",
               "predicted_bp_median", "predicted_bp_lo80", "predicted_bp_hi80"]
STR_ROLES = ("model", "quantile_lo", "quantile_med", "quantile_hi", "calibration", "spec")


def _opener(path, mode):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def _split_int(value):
    out = []
    for part in str(value).split("/"):
        try:
            out.append(int(part))
        except ValueError:
            out.append(-1)
    return out


def _gc(motif):
    motif = motif.upper()
    return (motif.count("G") + motif.count("C")) / len(motif) if motif else 0.0


def read_vcf(path):
    header, records = [], []
    with _opener(path, "rt") as handle:
        for line in handle:
            (header if line.startswith("#") else records).append(line.rstrip("\n"))
    if not header or not header[-1].startswith("#CHROM"):
        raise ValueError(f"{path}: not a VCF (missing #CHROM header line)")
    samples = header[-1].split("\t")[9:]
    parsed = []
    for record in records:
        fields = record.split("\t")
        info = dict(kv.split("=", 1) if "=" in kv else (kv, "") for kv in fields[7].split(";"))
        keys = fields[8].split(":")
        values = [dict(zip(keys, s.split(":"))) for s in fields[9:9 + len(samples)]]
        parsed.append((fields, info, values))
    return header, samples, parsed


def read_caller_table(path):
    """Parsed ExpansionHunter caller table -> the same in-memory structure as a single-sample VCF (GT absent)."""
    sample = Path(path).name.split(".")[0]
    header = ["##fileformat=VCFv4.2", f"##source=annStrata: converted from ExpansionHunter caller table {Path(path).name}",
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + sample]
    parsed = []
    with _opener(path, "rt") as handle:
        columns = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(columns)}
        for line in handle:
            p = line.rstrip("\n").split("\t")
            fields = [p[index["chrom"]], p[index["pos"]], ".", "N", ".", ".", p[index["filter"]],
                      f"END={p[index['end']]};REF={p[index['ref']]};RL={p[index['rl']]};RU={p[index['ru']]};VARID={p[index['repid']]};REPID={p[index['repid']]}",
                      "SO:REPCN:REPCI:ADSP:ADFL:ADIR:LC",
                      ":".join([p[index["so"]], p[index["repcn"]], p[index["repci"]], p[index["adsp"]], p[index["adfl"]], p[index["adir"]], p[index["lc"]]])]
            info = dict(kv.split("=", 1) for kv in fields[7].split(";"))
            values = [dict(zip(fields[8].split(":"), fields[9].split(":")))]
            parsed.append((fields, info, values))
    return header, [sample], parsed


def load_bundle(registry, root):
    """Four LightGBM boosters + calibration knots + spec. Returns (bundle dict, {role: path})."""
    entry = registry["models"]["str"]
    paths = {role: resolve_artifact(entry[role], root) for role in STR_ROLES}
    boosters = {role: load_booster(paths[role]) for role in ("model", "quantile_lo", "quantile_med", "quantile_hi")}
    for role, b in boosters.items():
        if b.feature_name() != F.FEATURES_SHORT:
            raise RuntimeError(f"STR {role}: feature order does not match the tool's feature definitions")
    with paths["calibration"].open(encoding="utf-8") as handle:
        cal = json.load(handle)
    with paths["spec"].open(encoding="utf-8") as handle:
        spec = json.load(handle)
    bundle = {"boosters": boosters, "cal_x": np.asarray(cal["x"], dtype=np.float64), "cal_y": np.asarray(cal["y"], dtype=np.float64), "spec": spec}
    return bundle, paths


def _predict(bundle, X, eh_bp, threads):
    """Calibrated error probability and (median, lo80, hi80) bp for one feature block."""
    B = bundle["boosters"]
    p = np.interp(B["model"].predict(X, num_threads=threads), bundle["cal_x"], bundle["cal_y"])
    q = np.sort(np.column_stack([B[r].predict(X, num_threads=threads) for r in ("quantile_lo", "quantile_med", "quantile_hi")]), axis=1)
    bp = np.rint(np.maximum(eh_bp, 1.0)[:, None] * 2.0 ** q).astype(int)
    return p, bp[:, 1], bp[:, 0], bp[:, 2]


def _features(rows, lc_med, lc_mad, rank):
    A = lambda k, j=None: np.array([d[k] if j is None else d[k][j] for d in rows])
    me, pa = (0, 1) if rank == 0 else (1, 0)
    return F.allele_features(A(5, me).astype(float), A(1).astype(float), A(2), A(6, me), A(7, me), A(8, me), A(9, me), A(10, me), A(11, me), A(3), A(4), lc_med, lc_mad, rank,
                             A(5, pa).astype(float), A(6, pa), A(7, pa), A(8, pa), A(9, pa), A(10, pa), A(11, pa))


def parse_call(fields, info, v):
    """One sample's call -> (status, row). status is 'diploid' or 'haploid' (scored) or the summary key of the reason it is not scored.
    row = (ru_len, ru_gc, lc, lowdepth, cn[2], so[2], adsp[2], adfl[2], adir[2], ci_lo[2], ci_hi[2]), shorter allele first.
    A haploid call fills both slots with the same values, as the training data preparation does."""
    ru = info.get("RU", "")
    if fields[0] in OUT_OF_SCOPE:
        return "out_of_scope_chrY_chrM", None
    if not ru:
        return "unscorable_missing_ru", None
    cn_text = v.get("REPCN", ".")
    if cn_text in (".", "./.", ""):
        return "unscorable_missing_repcn", None
    cns = _split_int(cn_text)
    if any(c < 0 for c in cns):
        return "unscorable_missing_repcn", None
    hap = len(cns) == 1
    if hap:
        cns = cns * 2
    if len(cns) != 2:
        return "unscorable_other", None
    try:
        lc = float(v.get("LC", "nan"))
    except ValueError:
        lc = np.nan
    lowdepth = 0 if fields[6] in ("PASS", ".") else 1
    so = v.get("SO", ".").split("/")
    adsp, adfl, adir = _split_int(v.get("ADSP", ".")), _split_int(v.get("ADFL", ".")), _split_int(v.get("ADIR", "."))
    cil, cih = [], []
    for c in v.get("REPCI", ".").split("/"):
        p = _split_int(c.replace("-", "/")) if "-" in c else [-1, -1]
        cil.append(p[0]); cih.append(p[1] if len(p) > 1 else -1)
    o = (1, 0) if cns[1] < cns[0] else (0, 1)
    pad = lambda x, m: ((x * 2) if hap else (x + [m] * 2))[:2]
    so2, adsp2, adfl2, adir2, cil2, cih2 = pad(so, "."), pad(adsp, -1), pad(adfl, -1), pad(adir, -1), pad(cil, -1), pad(cih, -1)
    return ("haploid" if hap else "diploid"), (len(ru), _gc(ru), lc, lowdepth, [cns[i] for i in o], [so2[i] for i in o], [adsp2[i] for i in o],
                                               [adfl2[i] for i in o], [adir2[i] for i in o], [cil2[i] for i in o], [cih2[i] for i in o])


def score_parsed(parsed, samples, bundle, threads):
    """Returns per-record per-sample [ASES, ASEL, ASLS, ASLL] strings, per-allele TSV rows and a summary."""
    out_fields = [[["."] * 4 for _ in samples] for _ in parsed]
    tsv_rows = []
    summary = dict(records=len(parsed), samples=len(samples), scored_diploid=0, scored_haploid=0,
                   unscorable_missing_repcn=0, unscorable_missing_ru=0, unscorable_other=0, out_of_scope_chrY_chrM=0)
    for si, sample in enumerate(samples):
        lcs = []
        for _, _, values in parsed:
            try:
                lcs.append(float(values[si].get("LC", ".")))
            except ValueError:
                pass
        lcs = np.asarray(lcs)
        lc_med = float(np.median(lcs)) if len(lcs) else 1.0
        lc_mad = float(np.median(np.abs(lcs - lc_med)) * 1.4826 + 1e-6) if len(lcs) else 1.0
        calls = []
        for ri, (fields, info, values) in enumerate(parsed):
            status, row = parse_call(fields, info, values[si])
            summary[status if row is None else f"scored_{status}"] += 1
            if row is not None:
                calls.append((ri,) + row)
        if not calls:
            continue
        rul = np.array([d[1] for d in calls], float); cs = np.array([d[5][0] for d in calls], float); cl = np.array([d[5][1] for d in calls], float)
        ps, ms, ls, hs = _predict(bundle, _features(calls, lc_med, lc_mad, 0), cs * rul, threads)
        pl, ml, ll, hl = _predict(bundle, _features(calls, lc_med, lc_mad, 1), cl * rul, threads)
        for j, d in enumerate(calls):
            out_fields[d[0]][si] = [f"{ps[j]:.3f}", f"{pl[j]:.3f}", f"{ms[j]},{ls[j]},{hs[j]}", f"{ml[j]},{ll[j]},{hl[j]}"]
            fields, info, _ = parsed[d[0]]
            base = [sample, fields[0], fields[1], info.get("REPID", info.get("VARID", ".")), info.get("RU", ".")]
            tsv_rows.append(base + ["short", d[6][0], int(cs[j]), int(cs[j] * rul[j]), f"{ps[j]:.4f}", ms[j], ls[j], hs[j]])
            tsv_rows.append(base + ["long", d[6][1], int(cl[j]), int(cl[j] * rul[j]), f"{pl[j]:.4f}", ml[j], ll[j], hl[j]])
    return out_fields, tsv_rows, summary


def write_vcf(path, header, parsed, out_fields, samples, model_id, summary):
    new_header = VCF_FORMAT_HEADER + [
        f'##annStrataCommandLine=<ID=annStrata,Version={__version__}-{model_id},AlphaT={ALPHA},FloorRU={FLOOR_RU},Band="long-read anchored",'
        f'Date="{time.strftime("%Y-%m-%d")}",Summary="{json.dumps(summary).replace(chr(34), "")}",CommandLine="{" ".join(sys.argv)}">']
    format_lines = [i for i, h in enumerate(header) if h.startswith("##FORMAT")]
    cut = (format_lines[-1] + 1) if format_lines else len(header) - 1
    header2 = header[:cut] + new_header + header[cut:]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _opener(path, "wt") as out:
        out.write("\n".join(header2) + "\n")
        for (fields, _, _), ofs in zip(parsed, out_fields):
            row = fields[:8] + [fields[8] + ":ASES:ASEL:ASLS:ASLL"] + [fields[9 + si] + ":" + ":".join(ofs[si]) for si in range(len(samples))]
            out.write("\t".join(row) + "\n")


def write_tsv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _opener(path, "wt") as out:
        out.write("\t".join(TSV_COLUMNS) + "\n")
        for row in rows:
            out.write("\t".join(str(x) for x in row) + "\n")


def score_str(args, registry, root):
    t0 = time.time()
    if not (args.out_vcf or args.out_tsv):
        raise ValueError("Specify --out-vcf and/or --out-tsv")
    bundle, paths = load_bundle(registry, root)
    model_id = registry["models"]["str"]["model"]["sha256"][:12]
    is_table = args.input.endswith((".tsv", ".tsv.gz"))
    header, samples, parsed = read_caller_table(args.input) if is_table else read_vcf(args.input)
    if not parsed:
        raise RuntimeError(f"No STR records were parsed from {args.input}")
    out_fields, tsv_rows, summary = score_parsed(parsed, samples, bundle, args.threads)
    if args.out_vcf:
        write_vcf(args.out_vcf, header, parsed, out_fields, samples, model_id, summary)
    if args.out_tsv:
        write_tsv(args.out_tsv, tsv_rows)
    summary["minutes"] = round((time.time() - t0) / 60, 2)
    spec = bundle["spec"]
    model_meta = {
        "family": "STR", "model_version": f"{__version__}-{model_id}",
        "endpoints": {
            "error_probability": f"ASES/ASEL and TSV error_probability = calibrated P(|EH length - long-read length| > max({FLOOR_RU} RU, {ALPHA} x long-read length)); higher is worse",
            "corrected_length": "ASLS/ASLL and TSV predicted_bp_* = predicted long-read length in bp: median, 10% and 90% quantiles",
        },
        "truth": spec.get("truth"), "training": spec.get("training_rows"), "scope": spec.get("scope"),
        "artifacts": [{"name": "str", "role": r, "path": str(p), "bytes": p.stat().st_size, "sha256": registry["models"]["str"][r].get("sha256")} for r, p in paths.items()],
    }
    outputs = [p for p in (args.out_vcf, args.out_tsv) if p]
    metadata = write_metadata(outputs[0], command="str", inputs=[args.input], model=model_meta, results=summary, include_input_sha256=args.hash_input)
    return summary, metadata
