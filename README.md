# annStrata

**Long-read-supervised reliability scores for short-read short tandem repeat (STR) and structural variant (SV) calls.**

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.12%2B-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

annStrata gives each STR allele and SV call in a short-read VCF the probability that it disagrees with the long-read
result, and each STR allele a corrected length. The models were trained on 274 genomes of the Human Pangenome Reference
Consortium (HPRC) with matched Illumina and PacBio HiFi data. Applying them needs only the short-read VCF and, for SVs,
the reference genome and public annotation files. Read alignments, long reads, and joint call sets are not needed.

**Outputs**

- **STR alleles** (ExpansionHunter or DRAGEN Repeat Genotyper VCF): a calibrated discordance probability, the probability
  that the called length differs from the long-read length by more than max(1 repeat unit, 0.5 × long-read length), and a
  corrected length, the predicted median of the long-read length with an 80% interval.
- **SV calls** (Manta VCF; Delly and LUMPY VCFs are also accepted): a calibrated discordance probability, the probability
  that the call has no matching event in a consensus of at least two of three long-read SV callers (Sniffles2, cuteSV,
  and Sawfish).

The values are added to the input VCF as FORMAT fields, existing fields are left unchanged, and every input record is
kept. No common threshold is applied, so a threshold can be chosen for each variant class and analysis.

---

## Requirements

- Python 3.12 or later (tested on Linux)
- numpy, pandas, and LightGBM 4.7 (installed with the package); pysam for SV VCF input
- For SV scoring from a VCF: the GRCh38 reference FASTA that the calls were made against and the annotation BEDs
  (see [resources/README.md](resources/README.md))

---

## Installation

**1. Clone the repository and install the package**

The repository includes the model weights (gzip-compressed LightGBM model files, 104 MB in total).

```bash
git clone https://github.com/joonan-lab/annStrata.git
cd annStrata
python3 -m venv annstrata-venv && source annstrata-venv/bin/activate
pip install -e ".[sv-vcf]"
annstrata self-test --checksum
```

Install in editable mode (`-e`) so that the package reads `models/` from the repository. `self-test --checksum` checks every model file against the SHA-256 in `models/registry.json` and reproduces frozen
predictions for a small STR and SV test set. To keep the models in another directory, set `ANNSTRATA_MODEL_ROOT` or pass
`--model-root`.

**2. For SV scoring from a VCF: download the annotation BEDs and the reference**

```bash
bash scripts/download_resources.sh   # GIAB stratifications v3.6 and UCSC RepeatMasker, prepared in resources/
```

The script checks each prepared file against the files used for training. The GRCh38 reference FASTA (3.2 GB) is
described in [resources/README.md](resources/README.md).

---

## Quick start

```bash
# STR: ExpansionHunter or DRAGEN Repeat Genotyper VCF
annstrata str sample.repeats.vcf.gz --out-vcf sample.annStrata.vcf.gz --out-tsv sample.annStrata.tsv.gz --threads 8

# SV: Manta VCF (Delly or LUMPY with --caller delly or --caller lumpy)
annstrata sv sample.manta.vcf.gz --fasta GRCh38_full_analysis_set_plus_decoy_hla.fa \
    --out-vcf sample.sv.annStrata.vcf.gz --out-tsv sample.sv.annStrata.tsv.gz

annstrata inspect          # model files, hashes, and environment
annstrata --help           # all options
```

An ExpansionHunter caller table (TSV) is also accepted by `annstrata str`, and a feature table written by
`annstrata/feature_builder.py` by `annstrata sv`. Every run also writes `<output>.metadata.json` with the software and model
versions, model hashes, and record counts.

Scoring one HPRC genome took 169 s for STRs and 35 s for SVs with eight threads on a server with two Intel Xeon Gold 6230
processors, with a peak memory of about 1.2 GB and 0.5 GB.

---

## Output

### STR

| Field | Description |
|---|---|
| `ASES`, `ASEL` | Discordance probability of the shorter and the longer allele (alleles ordered by repeat count) |
| `ASLS`, `ASLL` | Corrected length of the shorter and the longer allele in bp: `median,lo80,hi80` |
| TSV | `sample chrom pos repid ru allele so eh_repcn eh_bp error_probability predicted_bp_median predicted_bp_lo80 predicted_bp_hi80` |

Autosomes and chrX are scored. A haploid call (one repeat count, for example chrX of a male genome genotyped with its
sex) is scored as a homozygous call. chrY and chrM records and calls that cannot be scored carry `.`.

### SV

| Field | Description |
|---|---|
| `ASEP` | Discordance probability of the call |
| TSV | `sample chrom pos end svtype svlen filter error_probability` (scored calls only) |

DEL, INS, and DUP calls of 50 bp to 5 Mb on the autosomes and chrX are scored. Manta FILTER is not applied, so calls that
fail FILTER are scored like any other call. INV and BND records, chrY and chrM, other contigs, and calls outside 50 bp to
5 Mb carry `.` in the VCF.

---

## Interpreting the probabilities

- A higher probability means a higher chance that the allele or call disagrees with long-read data.
- The probabilities are calibrated for ExpansionHunter 5.0.0 and Manta 1.6.0 calls from Illumina genomes aligned to
  GRCh38. For other callers, the ranking carries over but the probabilities should be recalibrated before they are read
  as error rates.
- Calibration is weaker for STR alleles without spanning reads, so their probabilities are better used as rankings.
- Because the probabilities are calibrated, they can be summed to an expected number of discordant results or combined
  across the members of a family.

---

## Repository layout

| Path | Contents |
|---|---|
| `annstrata/` | Package: command-line interface, STR and SV scoring, feature builder, self-test |
| `models/` | Model weights (gzip-compressed LightGBM model files), registry, feature specifications, calibration maps, and feature importance |
| `resources/` | Annotation BEDs for SV scoring, created by `bash scripts/download_resources.sh` |
| `scripts/` | `download_resources.sh` |
| `tests/` | Test inputs and frozen predictions for `annstrata self-test` |

---

## Citation

If you use annStrata, please cite:

Yeom K, Kim W, Lee H, Hong H, Bong G, Han JH, Trost B, Cunial F, Lee E, Kim IB, Yoo HJ, An J-Y. annStrata:
long-read-supervised reliability assessment for cohort-scale analysis of short-read structural variants and tandem
repeats.

---

## License

MIT. See [LICENSE](LICENSE).
