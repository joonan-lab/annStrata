#!/usr/bin/env bash
# Download and prepare the annotation BEDs that `annstrata sv` reads when it scores an SV VCF:
#   GIAB genome stratifications v3.6 for GRCh38 (segmental duplications, low mappability, tandem repeats and
#   homopolymers, all difficult regions) and the UCSC hg38 RepeatMasker table (repClass/repFamily).
# Each file is reduced to the columns the feature builder reads, sorted, bgzip-compressed and tabix-indexed with pysam,
# and checked against the SHA-256 of the uncompressed BED used to train the SV model.
#
# usage: scripts/download_resources.sh [output directory]      (default: resources/ in the repository)
# needs: curl, awk, sort, and Python with pysam (pip install "annstrata[sv-vcf]")
set -euo pipefail

OUT=${1:-"$(cd "$(dirname "$0")/.." && pwd)/resources"}
PYTHON=${PYTHON:-python3}
GIAB=https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/genome-stratifications/v3.6/GRCh38@all
RMSK=https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz

"$PYTHON" -c "import pysam" 2>/dev/null || { echo "pysam is required (pip install 'annstrata[sv-vcf]')" >&2; exit 1; }
mkdir -p "$OUT/giab_prepared" "$OUT/repeatmasker"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# prepare <input .gz> <output .bed.gz> <awk program>: select columns, sort, bgzip and index
prepare() {
  local bed=${2%.gz}
  gzip -dc "$1" | awk -F'\t' -v OFS='\t' "$3" | LC_ALL=C sort -k1,1 -k2,2n > "$bed"
  "$PYTHON" -c 'import sys, pysam; pysam.tabix_index(sys.argv[1], preset="bed", force=True)' "$bed"
}

for entry in \
    segdups:SegmentalDuplications/GRCh38_segdups.bed.gz \
    lowmap:Mappability/GRCh38_lowmappabilityall.bed.gz \
    tandem:LowComplexity/GRCh38_AllTandemRepeatsandHomopolymers_slop5.bed.gz \
    difficult:Union/GRCh38_alldifficultregions.bed.gz; do
  name=${entry%%:*}
  echo "GIAB v3.6 ${entry#*:} -> giab_prepared/$name.bed.gz"
  curl -fsSL --retry 3 -o "$TMP/$name.bed.gz" "$GIAB/${entry#*:}"
  prepare "$TMP/$name.bed.gz" "$OUT/giab_prepared/$name.bed.gz" '!/^#/ {print $1, $2, $3}'
done

echo "UCSC hg38 rmsk.txt.gz -> repeatmasker/rmsk_class.bed.gz"
curl -fsSL --retry 3 -o "$TMP/rmsk.txt.gz" "$RMSK"
prepare "$TMP/rmsk.txt.gz" "$OUT/repeatmasker/rmsk_class.bed.gz" '{print $6, $7, $8, $12 "/" $13}'

# SHA-256 of the uncompressed BEDs used for model training
status=0
while read -r sum file; do
  got=$(gzip -dc "$OUT/$file.gz" | sha256sum | cut -d' ' -f1)
  if [ "$got" = "$sum" ]; then echo "OK        $file.gz"; else echo "MISMATCH  $file.gz (the source file has changed)"; status=1; fi
done <<'EOF'
3cfa219137b0abee326fff111b59345ff0767ef2ce4ec5b19823c5de8498caa8 giab_prepared/segdups.bed
f61c2a361f2494ba1acf299b8c2b38e9a6495f6c46873d0da9c468d936bc39cf giab_prepared/lowmap.bed
d5ddb46a57dc2477a8984726a10301a7959a358c5f73ab1a70d3760487197f8b giab_prepared/tandem.bed
98eb9d1c45b3e2f3d36a6e274883b20a353aea1376cedae818b8ed2a5fb24605 giab_prepared/difficult.bed
647a641f34e77710830ccd62db7b84db82ffac31c9a2c4c31cdacd2a07763b19 repeatmasker/rmsk_class.bed
EOF
exit $status
