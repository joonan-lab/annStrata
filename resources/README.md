# Resources for SV scoring

`annstrata sv` needs these files when it scores an SV VCF. STR scoring and SV scoring from a feature table do not.

## Annotation BEDs

Run `scripts/download_resources.sh` from the repository directory. It downloads the files below, keeps the columns the
feature builder reads, sorts, compresses, and indexes them, and checks each prepared file against the SHA-256 of the file
used for training.

| Prepared file | Source |
|---|---|
| `giab_prepared/segdups.bed.gz` | GIAB genome stratifications v3.6, GRCh38: `SegmentalDuplications/GRCh38_segdups.bed.gz` |
| `giab_prepared/lowmap.bed.gz` | GIAB v3.6, GRCh38: `Mappability/GRCh38_lowmappabilityall.bed.gz` |
| `giab_prepared/tandem.bed.gz` | GIAB v3.6, GRCh38: `LowComplexity/GRCh38_AllTandemRepeatsandHomopolymers_slop5.bed.gz` |
| `giab_prepared/difficult.bed.gz` | GIAB v3.6, GRCh38: `Union/GRCh38_alldifficultregions.bed.gz` |
| `repeatmasker/rmsk_class.bed.gz` | UCSC hg38 RepeatMasker table `rmsk.txt.gz` (chromosome, start, end, repClass/repFamily) |

GIAB: https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/genome-stratifications/v3.6/GRCh38@all/
UCSC: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz

`annstrata sv` reads them from this directory by default. Other locations can be given with `--giab-dir` and
`--repeatmasker`.

## Reference genome

The GC content, sequence entropy, and microhomology features are computed from the GRCh38 reference that the SV calls
were made against. The models were trained with the 1000 Genomes GRCh38 full analysis set with decoy and HLA sequences.

```bash
curl -LO https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa
curl -LO https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa.fai
```

| File | Bytes | SHA-256 |
|---|---|---|
| `GRCh38_full_analysis_set_plus_decoy_hla.fa` | 3,263,683,042 | `3b103f4742abfd54938fb0333e19ad067635c8eb86f1dbf0ce44b165c4292b50` |
| `GRCh38_full_analysis_set_plus_decoy_hla.fa.fai` | 154,196 | `f361f004bdae5ca68d632458b01a3848d02834ac7176f378e177344d148a6a6d` |

Pass it with `--fasta`.
