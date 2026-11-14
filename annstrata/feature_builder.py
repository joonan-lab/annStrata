#!/usr/bin/env python3
"""
annStrata feature builder (VCF + reference only, single sample).

Computes per-record features for SV calls (callers: manta, delly, lumpy) and STR calls
(callers: expansionhunter, gangstr). The SV model uses 33 of the SV columns (models/sv/spec.json).

Design constraints:
- Inputs are ONLY a short-read VCF + reference FASTA + static annotation BEDs.
  No BAM, no cohort, no long-read (long-read is used elsewhere for labeling only).
- Features are caller-common *concepts*; each caller is parsed by its own parser.
- `caller` is recorded for bookkeeping but is NOT emitted as a model feature.

Annotation BEDs must be sorted, bgzipped and tabix-indexed (see resources/README.md).

ExpansionHunter input is its flat (optionally gzipped) TSV, not a VCF — pass it to --vcf.

Parsing notes: pysam returns Number=2 fields (e.g. GangSTR REPCN/REPCI) as tuples, absent Flags
as False, and consumes INFO/END into rec.stop. The missing sentinel (-99999) lies outside the range
of every feature, including negative expansion_over_ref (contractions).

Usage:
    python feature_builder.py \
        --vcf sample.manta.vcf.gz --caller manta \
        --fasta GRCh38.fa \
        --giab-dir ../resources/giab_prepared \
        --repeatmasker ../resources/repeatmasker/rmsk_class.bed.gz \
        -o sample.manta.features.tsv
"""

import os
import sys
import math
import bisect
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import pysam

MISSING = -99999.0  # out-of-range sentinel for missing fields (paired with *_missing indicators).
# Must be outside every feature's real range: expansion_over_ref can legitimately be negative,
# so a small sentinel like -1 would collide with real contractions.
SV_CALLERS = {"manta", "delly", "lumpy"}
STR_CALLERS = {"expansionhunter", "gangstr"}
PRIMARY_CONTIGS = ({f"chr{c}" for c in list(range(1, 23)) + ["X", "Y", "M"]}
                   | {str(c) for c in list(range(1, 23)) + ["X", "Y", "MT", "M"]})

# Features that can legitimately be MISSING. Their `<feat>_missing` indicator is
# emitted ALWAYS (even if all-zero for a given caller) so every caller's output
# has an identical, fixed column schema — one trained model consumes any caller's
# converted VCF directly, no per-caller alignment needed.
SV_MISSING_INDICATORS = [
    "svlen_log", "cipos_width", "ciend_width", "vaf", "qual_norm", "gq",
    "local_depth", "gt_hom", "gc_min", "gc_max", "entropy_min", "microhom_max",
    "frac_span_repeat", "nn_log_dist",
]
STR_MISSING_INDICATORS = [
    "motif_len", "ref_copynum", "locus_depth", "gt_hom", "gt_repcn_max", "gt_repcn_min",
    "expansion_over_ref", "repci_width_max", "spanning_frac", "ref_tract_bp",
    "allele_vs_readlen", "motif_is_homopolymer", "gc_flank", "entropy_flank",
]


# ---------------------------------------------------------------------------
# Reference-sequence features
# ---------------------------------------------------------------------------
def gc_content(seq):
    if not seq:
        return MISSING
    seq = seq.upper()
    n = sum(1 for b in seq if b in "ACGT")
    if n == 0:
        return MISSING
    return sum(1 for b in seq if b in "GC") / n


def shannon_entropy(seq):
    if not seq:
        return MISSING
    seq = seq.upper()
    counts = defaultdict(int)
    for b in seq:
        if b in "ACGT":
            counts[b] += 1
    total = sum(counts.values())
    if total == 0:
        return MISSING
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def fetch(fasta, chrom, start, end):
    """0-based half-open fetch with clamping; returns '' on failure."""
    try:
        start = max(0, start)
        return fasta.fetch(chrom, start, end)
    except Exception:
        return ""


def gc_entropy_at(fasta, chrom, pos1, win):
    """GC and entropy in pos +/- win (pos is 1-based)."""
    seq = fetch(fasta, chrom, pos1 - 1 - win, pos1 + win)
    return gc_content(seq), shannon_entropy(seq)


def microhomology(fasta, chrom, pos1, end1, max_k=50):
    """
    Approximate microhomology between the two breakpoints of an intra-chromosomal
    SV: longest k (<=max_k) where the sequence adjacent to bp1 matches bp2.
    Returns MISSING for inter-chromosomal / undefined cases.
    """
    if end1 is None or end1 <= pos1:
        return MISSING
    left = fetch(fasta, chrom, pos1 - max_k, pos1 + max_k).upper()
    right = fetch(fasta, chrom, end1 - max_k, end1 + max_k).upper()
    if len(left) < 2 * max_k or len(right) < 2 * max_k:
        return MISSING
    k = 0
    while k < max_k and left[max_k + k] == right[max_k + k]:  # rightward match
        k += 1
    j = 0
    while j < max_k and left[max_k - 1 - j] == right[max_k - 1 - j]:  # leftward
        j += 1
    return float(max(k, j))


# ---------------------------------------------------------------------------
# Tabix annotation (binary overlap + RepeatMasker element class)
# ---------------------------------------------------------------------------
class Annotator:
    """Binary overlap against tabixed BEDs, with chr-naming fallback."""

    RMSK_ELEMENTS = {  # label prefix in rmsk_class.bed (repClass/repFamily) -> flag
        "SINE/Alu": "Alu",
        "LINE/L1": "L1",
        "Retroposon/SVA": "SVA",
        "LTR": "LTR",
    }

    def __init__(self, giab_dir=None, repeatmasker=None):
        self.tbx = {}
        self._cache = {}
        if giab_dir:
            for name in ("segdups", "lowmap", "tandem", "difficult"):
                p = os.path.join(giab_dir, f"{name}.bed.gz")
                if os.path.exists(p):
                    self.tbx[name] = pysam.TabixFile(p)
                else:
                    sys.stderr.write(f"[warn] missing GIAB bed: {p}\n")
        self.rmsk = pysam.TabixFile(repeatmasker) if repeatmasker and os.path.exists(repeatmasker) else None

    def _contigs(self, tbx, chrom):
        if chrom in tbx.contigs:
            return chrom
        alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
        return alt if alt in tbx.contigs else None

    def overlaps(self, name, chrom, pos1):
        """1 if 1-based pos overlaps any interval in bed `name`, else 0."""
        tbx = self.tbx.get(name)
        if tbx is None:
            return MISSING
        c = self._contigs(tbx, chrom)
        if c is None:
            return 0
        try:
            for _ in tbx.fetch(c, pos1 - 1, pos1):
                return 1
        except Exception:
            return 0
        return 0

    def _arrays(self, name, c):
        """Sorted, non-overlapping intervals of bed `name` on contig `c` as arrays (start, end, cumulative length), loaded once.
        Returns None when the contig's intervals are not sorted / disjoint (then the per-query tabix scan is used)."""
        key = (name, c)
        if key not in self._cache:
            starts, ends = [], []
            try:
                for row in self.tbx[name].fetch(c):
                    f = row.split("\t")
                    starts.append(int(f[1])); ends.append(int(f[2]))
            except Exception:
                starts = []
            S, E = np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)
            ok = len(S) > 0 and bool(np.all(np.diff(S) >= 0)) and bool(np.all(S[1:] >= E[:-1]))
            self._cache[key] = (S, E, np.concatenate([[0], np.cumsum(E - S)])) if ok else None
        return self._cache[key]

    def frac_overlap(self, name, chrom, start1, end1):
        """Fraction of [start1,end1] (1-based inclusive) covered by bed `name`."""
        tbx = self.tbx.get(name)
        if tbx is None or end1 is None or end1 < start1:
            return MISSING
        c = self._contigs(tbx, chrom)
        if c is None:
            return 0.0
        span = end1 - start1 + 1
        a, b = start1 - 1, end1                      # 0-based half-open query, as tabix fetch(c, start1 - 1, end1)
        arr = self._arrays(name, c)
        if arr is not None:                          # exact closed form on sorted disjoint intervals: O(log n) instead of a scan
            S, E, cum = arr
            i = int(np.searchsorted(E, a, side="right")); j = int(np.searchsorted(S, b, side="left"))
            if j <= i:
                return 0.0
            covered = int(cum[j] - cum[i]) - max(0, a - int(S[i])) - max(0, int(E[j - 1]) - b)
            return min(1.0, covered / span) if span > 0 else 0.0
        covered = 0
        try:
            for row in tbx.fetch(c, a, b):
                f = row.split("\t")
                s, e = int(f[1]), int(f[2])
                covered += max(0, min(end1, e) - max(start1 - 1, s))
        except Exception:
            return 0.0
        return min(1.0, covered / span) if span > 0 else 0.0

    def interval_len(self, name, chrom, pos1):
        """Length (end - start) of the interval of bed `name` containing 1-based pos1 (the interval `overlaps` finds), 0 if none."""
        tbx = self.tbx.get(name)
        if tbx is None:
            return MISSING
        c = self._contigs(tbx, chrom)
        if c is None:
            return 0
        arr = self._arrays(name, c)
        if arr is not None:                          # sorted disjoint intervals: binary search
            S, E, _ = arr
            j = int(np.searchsorted(S, pos1 - 1, side="right")) - 1
            return int(E[j] - S[j]) if j >= 0 and pos1 - 1 < E[j] else 0
        try:
            return max((int(row.split("\t")[2]) - int(row.split("\t")[1]) for row in tbx.fetch(c, pos1 - 1, pos1)), default=0)
        except Exception:
            return 0

    def rmsk_elements(self, chrom, pos1):
        """Return dict {Alu,L1,SVA,LTR -> 0/1} for the position."""
        flags = {"Alu": 0, "L1": 0, "SVA": 0, "LTR": 0}
        if self.rmsk is None:
            return {k: MISSING for k in flags}
        c = self._contigs(self.rmsk, chrom)
        if c is None:
            return flags
        try:
            for row in self.rmsk.fetch(c, pos1 - 1, pos1):
                label = row.split("\t")[3]
                for prefix, flag in self.RMSK_ELEMENTS.items():
                    if label.startswith(prefix):
                        flags[flag] = 1
        except Exception:
            pass
        return flags


def agg_either_both(a, b):
    """Order-invariant aggregation for the two breakpoints."""
    if a == MISSING or b == MISSING:
        v = a if b == MISSING else b
        return v, v
    return (1 if (a or b) else 0), (1 if (a and b) else 0)


# ---------------------------------------------------------------------------
# Small helpers for VCF field access
# ---------------------------------------------------------------------------
def info(rec, key, default=None):
    try:
        return rec.info[key]
    except Exception:
        return default


SAMPLE_INDEX = 0   # column of the sample being scored in a multi-sample VCF (set from --sample)


def fmt(rec, key, default=None):
    try:
        return rec.samples[SAMPLE_INDEX][key]
    except Exception:
        return default


def is_pass(rec):
    fk = list(rec.filter.keys())
    return 1 if (not fk or fk == ["PASS"] or fk == ["."]) else 0


def gt_is_hom_alt(rec):
    gt = fmt(rec, "GT")
    if not gt or any(a is None for a in gt):
        return MISSING
    alleles = [a for a in gt]
    return 1 if all(a == alleles[0] and a > 0 for a in alleles) else 0


def first(x, default=MISSING):
    """Coerce a possibly-tuple INFO/FORMAT value to a scalar number."""
    if x is None:
        return default
    if isinstance(x, (tuple, list)):
        x = x[0] if x else default
    try:
        return float(x)
    except Exception:
        return default


def width(ci):
    if not ci or not isinstance(ci, (tuple, list)) or len(ci) < 2:
        return MISSING
    try:
        return abs(float(ci[1]) - float(ci[0]))
    except Exception:
        return MISSING


def norm_svtype(rec):
    st = info(rec, "SVTYPE")
    if st is None:
        alt = str(rec.alts[0]) if rec.alts else ""
        st = alt.strip("<>").split(":")[0] if alt.startswith("<") else "BND"
    st = str(st).upper().split(":")[0]
    if st in ("TRA", "CTX"):
        st = "BND"
    if st not in ("DEL", "DUP", "INS", "INV", "BND"):
        st = "BND"
    return st


# ---------------------------------------------------------------------------
# Per-caller SV parsers  ->  normalized concept dict
# ---------------------------------------------------------------------------
def parse_sv_common(rec):
    st = norm_svtype(rec)
    chrom = rec.chrom
    pos = rec.pos
    # pysam consumes INFO/END into rec.stop; meaningful only for spanned SVs.
    # BND/INS are annotated at their primary breakend only (bp2 = bp1 via end=None).
    end = rec.stop if st in ("DEL", "DUP", "INV") else None
    return {
        "chrom": chrom, "pos": pos, "end": end, "chrom2": chrom,
        "svtype": st,
        "is_pass": is_pass(rec),
        "cipos_width": width(info(rec, "CIPOS") or info(rec, "CIPOS95")),
        "ciend_width": width(info(rec, "CIEND") or info(rec, "CIEND95")),
        "is_imprecise": 1 if ("IMPRECISE" in rec.info) else 0,
        "gt_hom": gt_is_hom_alt(rec),
        "svlen_raw": info(rec, "SVLEN"),
    }


def parse_manta(rec):
    d = parse_sv_common(rec)
    pr = fmt(rec, "PR") or (None, None)
    sr = fmt(rec, "SR") or (None, None)
    pr_ref, pr_alt = (first(pr[0], 0), first(pr[1], 0)) if len(pr) == 2 else (0, 0)
    sr_ref, sr_alt = (first(sr[0], 0), first(sr[1], 0)) if len(sr) == 2 else (0, 0)
    tot = pr_ref + pr_alt + sr_ref + sr_alt
    d.update({
        "pe_support": pr_alt, "sr_support": sr_alt, "total_support": pr_alt + sr_alt,
        "vaf": (pr_alt + sr_alt) / tot if tot > 0 else MISSING,
        "gq": first(fmt(rec, "GQ")), "qual_norm": first(rec.qual),
        "local_depth": (pr_ref + pr_alt) or first(info(rec, "BND_DEPTH")),
    })
    return d


def parse_delly(rec):
    d = parse_sv_common(rec)
    dr, dv = first(fmt(rec, "DR"), 0), first(fmt(rec, "DV"), 0)
    rr, rv = first(fmt(rec, "RR"), 0), first(fmt(rec, "RV"), 0)
    tot = dr + dv + rr + rv
    if d["svlen_raw"] is None and d["end"] is not None:  # v0.7 has no SVLEN
        d["svlen_raw"] = d["end"] - d["pos"]
    d.update({
        "pe_support": dv, "sr_support": rv, "total_support": dv + rv,
        "vaf": (dv + rv) / tot if tot > 0 else MISSING,
        "gq": first(fmt(rec, "GQ")), "qual_norm": first(rec.qual),
        "local_depth": dr + dv,
    })
    return d


def parse_lumpy(rec):
    d = parse_sv_common(rec)
    ao, ro = first(fmt(rec, "AO"), 0), first(fmt(rec, "RO"), 0)
    ab = fmt(rec, "AB")
    # smoove/LUMPY put SU/PE/SR in INFO (site-level), not FORMAT; fall back to FORMAT for other dialects
    pe = info(rec, "PE"); pe = first(pe) if pe is not None else first(fmt(rec, "PE"), 0)
    sr = info(rec, "SR"); sr = first(sr) if sr is not None else first(fmt(rec, "SR"), 0)
    su = info(rec, "SU"); su = first(su) if su is not None else first(fmt(rec, "SU"), 0)
    d.update({
        "pe_support": pe, "sr_support": sr, "total_support": su,
        "vaf": first(ab) if ab is not None else ((ao / (ao + ro)) if (ao + ro) > 0 else MISSING),
        "gq": first(fmt(rec, "GQ")), "qual_norm": first(fmt(rec, "SQ")),
        "local_depth": first(fmt(rec, "DP")),
    })
    return d


SV_PARSERS = {"manta": parse_manta, "delly": parse_delly, "lumpy": parse_lumpy}


# ---------------------------------------------------------------------------
# Per-caller STR parsers
# ---------------------------------------------------------------------------
def _split_pair(val, sep):
    if val is None:
        return []
    if isinstance(val, (tuple, list)):  # pysam returns Number=2 fields (e.g. GangSTR REPCN) as tuples
        out = []
        for x in val:
            try:
                out.append(float(x))
            except Exception:
                pass
        return out
    s = str(val)
    for d in sep:
        s = s.replace(d, "|")
    out = []
    for tok in s.split("|"):
        try:
            out.append(float(tok))
        except Exception:
            pass
    return out


def parse_eh(rec):
    ru = info(rec, "RU") or ""
    repcn = _split_pair(fmt(rec, "REPCN"), "/")
    ref_cn = first(info(rec, "REF"))
    adsp = sum(_split_pair(fmt(rec, "ADSP"), "/"))
    adfl = sum(_split_pair(fmt(rec, "ADFL"), "/"))
    adir = sum(_split_pair(fmt(rec, "ADIR"), "/"))
    return {
        "chrom": rec.chrom, "pos": rec.pos, "end": rec.stop,
        "is_pass": is_pass(rec), "motif_len": float(len(ru)) if ru else first(info(rec, "RL")),
        "ref_copynum": ref_cn,
        "repcn": repcn, "repci_raw": fmt(rec, "REPCI"),
        "spanning_reads": adsp, "flanking_reads": adfl, "inrepeat_reads": adir,
        "locus_depth": first(fmt(rec, "LC")), "gt_hom": gt_is_hom_alt(rec),
        "qual_post": first(rec.qual), "ref_tract_bp": first(info(rec, "RL")),
        "ru": ru,
    }


def parse_gangstr(rec):
    ru = info(rec, "RU") or ""
    period = first(info(rec, "PERIOD"))
    repcn = _split_pair(fmt(rec, "REPCN"), ",")
    ref_cn = first(info(rec, "REF"))
    rc = _split_pair(fmt(rec, "RC"), ",")  # enclosing,spanning,FRR,bounding
    enclosing, spanning, frr, bounding = (rc + [0, 0, 0, 0])[:4]
    return {
        "chrom": rec.chrom, "pos": rec.pos, "end": rec.stop,
        "is_pass": is_pass(rec), "motif_len": period if period != MISSING else float(len(ru)),
        "ref_copynum": ref_cn,
        "repcn": repcn, "repci_raw": fmt(rec, "REPCI"),
        "spanning_reads": enclosing + spanning, "flanking_reads": bounding, "inrepeat_reads": frr,
        "locus_depth": first(fmt(rec, "DP")), "gt_hom": gt_is_hom_alt(rec),
        "qual_post": first(fmt(rec, "Q")),
        "ref_tract_bp": (ref_cn * period) if (ref_cn != MISSING and period != MISSING) else MISSING,
        "ru": ru,
    }


def _num(x, default=MISSING):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return default if v != v else v  # NaN guard
    except Exception:
        return default


def parse_eh_tsv(row):
    """One row of an ExpansionHunter flat TSV:
    chrom,pos,end,filter,repid,ru,rl,ref,repcn,repci,adsp,adfl,adir,lc,so"""
    ru = str(row.get("ru") or "")
    repcn = _split_pair(row.get("repcn"), "/")
    ref_cn = _num(row.get("ref"))
    rl = _num(row.get("rl"))
    adsp = sum(_split_pair(row.get("adsp"), "/"))
    adfl = sum(_split_pair(row.get("adfl"), "/"))
    adir = sum(_split_pair(row.get("adir"), "/"))
    gt_hom = MISSING
    if len(repcn) >= 2:  # hom-ALT = both alleles equal and differ from reference
        gt_hom = 1 if (repcn[0] == repcn[1] and repcn[0] != ref_cn) else 0
    return {
        "chrom": str(row["chrom"]), "pos": int(float(row["pos"])), "end": _num(row.get("end")),
        "is_pass": 1 if str(row.get("filter", "")).upper() == "PASS" else 0,
        "motif_len": float(len(ru)) if ru else rl,
        "ref_copynum": ref_cn,
        "repcn": repcn, "repci_raw": row.get("repci"),
        "spanning_reads": adsp, "flanking_reads": adfl, "inrepeat_reads": adir,
        "locus_depth": _num(row.get("lc")), "gt_hom": gt_hom,
        "qual_post": MISSING,  # EH TSV carries no site quality
        "ref_tract_bp": rl, "ru": ru,
    }


STR_PARSERS = {"expansionhunter": parse_eh, "gangstr": parse_gangstr}


def repci_width_max(repci_raw):
    """Max allele CI width. EH: '2-2/10-10' (str); GangSTR: ('1-2','2-2') (pysam tuple)."""
    if repci_raw is None:
        return MISSING
    if isinstance(repci_raw, (tuple, list)):
        alleles = [str(x) for x in repci_raw]
    else:
        alleles = str(repci_raw).replace("/", ",").split(",")
    best = MISSING
    for allele in alleles:
        if "-" in allele:
            try:
                parts = allele.split("-")
                w = abs(float(parts[1]) - float(parts[0]))
                best = w if best == MISSING else max(best, w)
            except Exception:
                pass
    return best


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------
def sv_features(d, ann, fasta, win):
    chrom, pos, end = d["chrom"], d["pos"], d["end"]
    chrom2, end2 = d["chrom2"], (end if end is not None else pos)
    st = d["svtype"]
    svlen = first(d["svlen_raw"])
    f = {
        "variant_ID": f"{chrom}:{pos}:{st}:{end}",
        "is_pass": d["is_pass"],
        "svtype_DEL": int(st == "DEL"), "svtype_DUP": int(st == "DUP"),
        "svtype_INS": int(st == "INS"), "svtype_INV": int(st == "INV"),
        "svtype_BND": int(st == "BND"),
        "svlen_log": math.log10(abs(svlen) + 1) if svlen != MISSING else MISSING,
        "cipos_width": d["cipos_width"], "ciend_width": d["ciend_width"],
        "is_imprecise": d["is_imprecise"],
        "pe_support": d["pe_support"], "sr_support": d["sr_support"],
        "total_support": d["total_support"], "vaf": d["vaf"],
        "gt_hom": d["gt_hom"], "gq": d["gq"], "qual_norm": d["qual_norm"],
        "local_depth": d["local_depth"],
    }
    # reference sequence context at both breakpoints
    gc1, e1 = gc_entropy_at(fasta, chrom, pos, win)
    gc2, e2 = gc_entropy_at(fasta, chrom2, end2, win)
    f["gc_min"], f["gc_max"] = (min(gc1, gc2), max(gc1, gc2)) if MISSING not in (gc1, gc2) else (MISSING, MISSING)
    f["entropy_min"] = min(e1, e2) if MISSING not in (e1, e2) else MISSING
    f["microhom_max"] = microhomology(fasta, chrom, pos, end if chrom2 == chrom else None)
    # GIAB binary overlap, both breakpoints
    for name, key in (("segdups", "segdup"), ("difficult", "difficult")):
        ei, bo = agg_either_both(ann.overlaps(name, chrom, pos), ann.overlaps(name, chrom2, end2))
        f[f"in_{key}_either"], f[f"in_{key}_both"] = ei, bo
    for name, key in (("lowmap", "lowmap"), ("tandem", "tandem")):
        ei, _ = agg_either_both(ann.overlaps(name, chrom, pos), ann.overlaps(name, chrom2, end2))
        f[f"in_{key}_either"] = ei
    # RepeatMasker element class, either breakpoint
    r1 = ann.rmsk_elements(chrom, pos)
    r2 = ann.rmsk_elements(chrom2, end2)
    for elt in ("Alu", "L1", "SVA", "LTR"):
        ei, _ = agg_either_both(r1[elt], r2[elt])
        f[f"in_{elt}_either"] = ei
    # fraction of the SV interval covered by repeats (intra-chrom interval SVs only)
    if st in ("DEL", "DUP", "INV") and end is not None and chrom2 == chrom:
        f["frac_span_repeat"] = max(ann.frac_overlap("tandem", chrom, pos, end),
                                    ann.frac_overlap("segdups", chrom, pos, end))
    else:
        f["frac_span_repeat"] = MISSING
    # neighbor density (SV only) — precomputed onto d by compute_clustering()
    f["n_neighbors"] = d.get("n_neighbors", 0)
    f["nn_log_dist"] = d.get("nn_log_dist", MISSING)
    # length (bp) of the GIAB tandem-repeat interval containing a breakpoint, the larger of the two (INS/BND: the one position);
    # 0 = not in a tandem repeat, so tr_span_len > 0 is exactly in_tandem_either
    t1, t2 = ann.interval_len("tandem", chrom, pos), ann.interval_len("tandem", chrom2, end2)
    f["tr_span_len"] = max(t1, t2) if MISSING not in (t1, t2) else MISSING
    return f


def str_features(d, ann, fasta, win, read_len):
    chrom, pos = d["chrom"], d["pos"]
    repcn = d["repcn"] or []
    cn_max = max(repcn) if repcn else MISSING
    cn_min = min(repcn) if repcn else MISSING
    ref_cn = d["ref_copynum"]
    motif = d["motif_len"]
    f = {
        "variant_ID": f"{chrom}:{pos}:{info_end(d)}",
        "is_pass": d["is_pass"], "motif_len": motif, "ref_copynum": ref_cn,
        "gt_repcn_max": cn_max, "gt_repcn_min": cn_min,
        "expansion_over_ref": (cn_max - ref_cn) if MISSING not in (cn_max, ref_cn) else MISSING,
        "repci_width_max": repci_width_max(d["repci_raw"]),
        "spanning_reads": d["spanning_reads"], "flanking_reads": d["flanking_reads"],
        "inrepeat_reads": d["inrepeat_reads"],
        "locus_depth": d["locus_depth"], "gt_hom": d["gt_hom"],
        # no qual_post: EH does not emit it, so it would only encode caller identity
        "ref_tract_bp": d["ref_tract_bp"],
    }
    tot = d["spanning_reads"] + d["flanking_reads"] + d["inrepeat_reads"]
    f["spanning_frac"] = d["spanning_reads"] / tot if tot > 0 else MISSING
    f["allele_vs_readlen"] = (cn_max * motif / read_len) if MISSING not in (cn_max, motif) else MISSING
    f["motif_is_homopolymer"] = int(motif == 1) if motif != MISSING else MISSING
    gc, ent = gc_entropy_at(fasta, chrom, pos, win)
    f["gc_flank"], f["entropy_flank"] = gc, ent
    f["in_segdup"] = ann.overlaps("segdups", chrom, pos)
    f["in_difficult"] = ann.overlaps("difficult", chrom, pos)
    f["flank_lowmap"] = ann.overlaps("lowmap", chrom, pos)
    return f


def info_end(d):
    return int(d["end"]) if d.get("end") is not None else d["pos"]


# ---------------------------------------------------------------------------
# Clustering (SV) — within-callset neighbor density
# ---------------------------------------------------------------------------
def compute_clustering(parsed, radius):
    """Set on each parsed SV dict:
      nn_log_dist  = log10(distance to nearest other call + 1), UNCAPPED (isolation).
      n_neighbors  = number of other calls within +/-radius.
    SV calls are sparse (median nearest neighbor ~5-90 kb), so radius must be SV-scale
    (default 100 kb), not the 1 kb used for dense small variants. Vectorized per chrom."""
    by_chrom = defaultdict(list)
    for j, d in enumerate(parsed):
        by_chrom[d["chrom"]].append((d["pos"], j))
    for items in by_chrom.values():
        items.sort()
        pos = np.array([p for p, _ in items])
        n = len(pos)
        for k, (_, j) in enumerate(items):
            if n < 2:
                parsed[j]["nn_log_dist"], parsed[j]["n_neighbors"] = MISSING, 0
                continue
            p = pos[k]
            nearest = min((p - pos[k - 1]) if k > 0 else float("inf"),
                          (pos[k + 1] - p) if k < n - 1 else float("inf"))
            parsed[j]["nn_log_dist"] = math.log10(nearest + 1)
            lo = int(np.searchsorted(pos, p - radius, "left"))
            hi = int(np.searchsorted(pos, p + radius, "right"))
            parsed[j]["n_neighbors"] = hi - lo - 1  # exclude self


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="annStrata feature builder")
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--caller", required=True,
                    choices=sorted(SV_CALLERS | STR_CALLERS))
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--giab-dir", default=None, help="dir with segdups/lowmap/tandem/difficult .bed.gz (tabixed)")
    ap.add_argument("--repeatmasker", default=None, help="tabixed rmsk_class.bed.gz")
    ap.add_argument("--win", type=int, default=50, help="GC/entropy window (+/- bp)")
    ap.add_argument("--neighbor-radius", type=int, default=100000,
                    help="SV clustering radius for n_neighbors (+/- bp). Default 100kb — SV calls are "
                         "sparse (median nearest ~5-90kb); 1kb is for dense small variants.")
    ap.add_argument("--read-len", type=int, default=150, help="short-read length (STR spanning feasibility)")
    ap.add_argument("--primary-only", dest="primary_only", action="store_true", default=True,
                    help="keep only primary-assembly contigs chr1-22,X,Y,M (default on)")
    ap.add_argument("--all-contigs", dest="primary_only", action="store_false",
                    help="include ALT/decoy/HLA contigs (off by default)")
    ap.add_argument("--str-drop-homref", action="store_true",
                    help="(STR) drop hom-ref 0/0 genotype loci (catalog non-variants)")
    ap.add_argument("--sample", default=None,
                    help="sample id (default: auto from VCF's single sample, or EH-TSV filename prefix). "
                         "Emitted as a `sample` column — the label join key with the truth set.")
    ap.add_argument("--missing-indicators", action="store_true",
                    help="also emit <feat>_missing 0/1 columns. OFF by default: redundant for tree "
                         "models (the -99999 sentinel is already split-separable). Turn on for linear/NN models.")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    variant_class = "SV" if args.caller in SV_CALLERS else "STR"
    fasta = pysam.FastaFile(args.fasta)
    ann = Annotator(args.giab_dir, args.repeatmasker)

    eh_tsv = (args.caller == "expansionhunter")  # EH ships a flat (gzipped) TSV, not a VCF
    if eh_tsv:
        with open(args.vcf, "rb") as fh:
            comp = "gzip" if fh.read(2) == b"\x1f\x8b" else None
        records = pd.read_csv(args.vcf, sep="\t", dtype=str, compression=comp).to_dict("records")
        sample = args.sample or os.path.basename(args.vcf).split(".")[0]
        get_chrom = lambda r: str(r["chrom"])
        def is_homref(r):
            cn, ref = _split_pair(r.get("repcn"), "/"), _num(r.get("ref"))
            return bool(cn) and all(x == ref for x in cn)
    else:
        vf = pysam.VariantFile(args.vcf)
        hdr = list(vf.header.samples)
        sample = args.sample or (hdr[0] if len(hdr) == 1 else None)
        if sample is None:
            sys.exit(f"[error] --sample required: VCF has {len(hdr)} samples {hdr}")
        if sample not in hdr:
            sys.exit(f"[error] sample {sample} not in VCF samples {hdr}")
        global SAMPLE_INDEX
        SAMPLE_INDEX = hdr.index(sample)   # FORMAT fields (GT, PR, SR, GQ, ...) are read from this sample's column
        records = list(vf)
        get_chrom = lambda r: r.chrom
        is_homref = lambda r: not (set(fmt(r, "GT") or ()) - {0})
    sys.stderr.write(f"[info] sample={sample}\n")

    n_raw = len(records)
    indices = list(range(n_raw))   # position of each record in the input (all records); lets the scorer write results back into the VCF
    if args.primary_only:
        keep = [i for i, r in enumerate(records) if get_chrom(r) in PRIMARY_CONTIGS]
        records, indices = [records[i] for i in keep], [indices[i] for i in keep]
        sys.stderr.write(f"[info] primary-only: dropped {n_raw - len(records):,} non-primary-contig records\n")
    if variant_class == "STR" and args.str_drop_homref:
        before = len(records)
        keep = [i for i, r in enumerate(records) if not is_homref(r)]
        records, indices = [records[i] for i in keep], [indices[i] for i in keep]
        sys.stderr.write(f"[info] str-drop-homref: dropped {before - len(records):,} hom-ref loci\n")
    sys.stderr.write(f"[info] {len(records):,} records to process | caller={args.caller} class={variant_class}\n")

    rows = []
    if variant_class == "SV":
        parser = SV_PARSERS[args.caller]
        parsed = [parser(r) for r in records]
        compute_clustering(parsed, args.neighbor_radius)
        for r, i, d in zip(records, indices, parsed):
            f = sv_features(d, ann, fasta, args.win)
            f["caller"] = args.caller
            f["vcf_index"] = i
            f["filter"] = ";".join(r.filter.keys()) or "PASS"
            rows.append(f)
    else:
        parser = parse_eh_tsv if eh_tsv else STR_PARSERS[args.caller]
        for r in records:
            d = parser(r)
            f = str_features(d, ann, fasta, args.win, args.read_len)
            f["caller"] = args.caller
            rows.append(f)

    out = pd.DataFrame(rows)
    out["sample"] = sample
    # Missingness is carried by the -99999 sentinel in each feature (trees split on it
    # directly). Optional explicit indicators (fixed list -> stable schema) for linear/NN.
    if args.missing_indicators:
        indicators = SV_MISSING_INDICATORS if variant_class == "SV" else STR_MISSING_INDICATORS
        for col in indicators:
            out[f"{col}_missing"] = (out[col] == MISSING).astype(int) if col in out.columns else 0
    # meta (label join key) first: sample, caller, variant_ID — NOT model features
    meta = [c for c in ("sample", "caller", "variant_ID", "vcf_index", "filter") if c in out.columns]
    out = out[meta + [c for c in out.columns if c not in meta]]
    out.to_csv(args.output, sep="\t", index=False)
    sys.stderr.write(f"[info] wrote {len(out):,} rows x {out.shape[1]} cols -> {args.output}\n")


if __name__ == "__main__":
    main()
