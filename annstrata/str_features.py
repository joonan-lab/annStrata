"""Feature construction for the STR model
(42 columns, in the order of models/str/spec.json). Inputs come from one ExpansionHunter/DRAGEN VCF record plus the
sample-level LC median/MAD. Partner arrays may be NaN (single-allele call)."""
import numpy as np

SO_CATS = ["SPANNING", "FLANKING", "INREPEAT", "."]
FEATURES_SHORT = ["eh_cn", "eh_bp", "rank", "so=SPANNING", "so=FLANKING", "so=INREPEAT", "so=.", "adsp", "adfl", "adir", "ci_lo_cn", "ci_hi_cn", "ci_w_cn", "ci_w_bp", "ci_w_rel",
                  "lc", "lc_rel_med", "lc_z_mad", "lowdepth", "ru_len", "ru_gc", "partner_cn", "partner_bp", "pso=SPANNING", "pso=FLANKING", "pso=INREPEAT", "pso=.",
                  "partner_adsp", "partner_adfl", "partner_adir", "partner_ci_w_cn", "bp_diff", "bp_log_ratio", "gt_hom", "sp_frac", "adir_pos", "adfl_pos", "hom_adir", "hom_adfl",
                  "adsp_over_lc", "support_total", "partner_sp_frac"]
def allele_features(eh_cn, ru_len, ru_gc, so, adsp, adfl, adir, ci_lo, ci_hi, lc, lowdepth, lc_med, lc_mad, rank,
                    p_cn, p_so, p_adsp, p_adfl, p_adir, p_ci_lo, p_ci_hi):
    """Vectorised 42-feature block for one slot. Partner arrays may contain NaN (hemizygous call) → NaN features."""
    eh_cn = eh_cn.astype(np.float32); ru = ru_len.astype(np.float32); eh_bp = eh_cn * ru
    adsp, adfl, adir = adsp.astype(np.float32), adfl.astype(np.float32), adir.astype(np.float32)
    ci_lo, ci_hi = ci_lo.astype(np.float32), ci_hi.astype(np.float32); ci_w = ci_hi - ci_lo; lc = lc.astype(np.float32)
    p_cn = p_cn.astype(np.float32); p_bp = p_cn * ru
    p_adsp, p_adfl, p_adir = p_adsp.astype(np.float32), p_adfl.astype(np.float32), p_adir.astype(np.float32); p_ci_w = (p_ci_hi - p_ci_lo).astype(np.float32)
    hom = (eh_cn == p_cn).astype(np.float32); hom[np.isnan(p_cn)] = np.nan
    tot = adsp + adfl + adir; p_tot = p_adsp + p_adfl + p_adir
    def onehot(codes):
        oh = np.zeros((len(codes), 4), dtype=np.float32); v = codes >= 0; oh[np.flatnonzero(v), codes[v]] = 1.0; return oh
    so_c = np.array([SO_CATS.index(s) if s in SO_CATS else -1 for s in so]); pso_c = np.array([SO_CATS.index(s) if s in SO_CATS else -1 for s in p_so])
    X = np.column_stack([
        eh_cn, eh_bp, np.full(len(eh_cn), float(rank), np.float32), onehot(so_c), adsp, adfl, adir, ci_lo, ci_hi, ci_w, ci_w * ru, ci_w / (eh_cn + 1),
        lc, lc / lc_med, (lc - lc_med) / lc_mad, lowdepth.astype(np.float32), ru, ru_gc.astype(np.float32), p_cn, p_bp, onehot(pso_c),
        p_adsp, p_adfl, p_adir, p_ci_w, p_bp - eh_bp, np.log2((p_bp + ru) / (eh_bp + ru)), hom, np.where(tot > 0, adsp / np.maximum(tot, 1), 0.0),
        (adir > 0).astype(np.float32), (adfl > 0).astype(np.float32), hom * (adir > 0), hom * (adfl > 0), adsp / np.maximum(lc, 0.1), tot,
        np.where(p_tot > 0, p_adsp / np.maximum(p_tot, 1), 0.0)]).astype(np.float32)
    assert X.shape[1] == len(FEATURES_SHORT)
    return X


