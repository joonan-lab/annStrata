import argparse
import json
import sys

from . import __version__
from .io_utils import dependency_versions
from .registry import load_registry, model_root, resolve_artifact, verify_artifact
from .selftest import run_selftest
from .str_score import score_str
from .sv_score import score_sv

ROLES = ("model", "quantile_lo", "quantile_med", "quantile_hi", "calibration", "spec")


def _add_model_root(parser):
    parser.add_argument(
        "--model-root",
        help="Directory containing registry.json, str/ and sv/ (default: ANNSTRATA_MODEL_ROOT or the package's models/)",
    )


def parser():
    main = argparse.ArgumentParser(
        prog="annstrata",
        description="Reliability scoring for short-read STR and SV calls with long-read-guided models: per-allele calibrated error probability and corrected long-read length (STR), per-call calibrated discordance probability (SV).",
    )
    main.add_argument("--version", action="version", version=f"annstrata {__version__}")
    sub = main.add_subparsers(dest="command", required=True)

    str_parser = sub.add_parser("str", help="Score ExpansionHunter/DRAGEN STR calls per allele (error probability + corrected length)")
    str_parser.add_argument("input", help="ExpansionHunter/DRAGEN Repeat Genotyper VCF[.gz] (single- or multi-sample) or parsed caller table TSV[.gz]")
    str_parser.add_argument("--out-vcf", help="Input VCF with ASES/ASEL/ASLS/ASLL FORMAT fields appended (every record kept)")
    str_parser.add_argument("--out-tsv", help="Per-allele TSV: error_probability and predicted_bp_median/lo80/hi80")
    str_parser.add_argument("--threads", type=int, default=8)
    str_parser.add_argument("--hash-input", action="store_true", help="Record input SHA-256 in metadata")
    _add_model_root(str_parser)

    sv_parser = sub.add_parser("sv", help="Score short-read SV calls per call (probability of discordance with a long-read 2-of-3 consensus)")
    sv_parser.add_argument("input", help="Single-sample SV VCF[.gz] (Manta; Delly / LUMPY with --caller) or a feature table TSV[.gz] written by feature_builder.py")
    sv_parser.add_argument("--out-vcf", help="Input VCF with the ASEP FORMAT field appended (every record kept; VCF input only)")
    sv_parser.add_argument("--out-tsv", help="Per-call TSV: sample chrom pos end svtype svlen filter error_probability (scored calls only)")
    sv_parser.add_argument("--caller", choices=["manta", "delly", "lumpy"], default="manta")
    sv_parser.add_argument("--fasta", help="GRCh38 FASTA with .fai (the build the VCF was called against; not bundled, see resources/README.md)")
    sv_parser.add_argument("--giab-dir", help="Directory with tabixed segdups/lowmap/tandem/difficult BEDs (default: resources/giab_prepared; see resources/README.md)")
    sv_parser.add_argument("--repeatmasker", help="Tabixed RepeatMasker class BED (default: resources/repeatmasker/rmsk_class.bed.gz; see resources/README.md)")
    sv_parser.add_argument("--sample", help="Sample to score in a multi-sample VCF (default: the single sample)")
    sv_parser.add_argument("--features-out", help="Keep the generated feature table (TSV)")
    sv_parser.add_argument("--neighbor-radius", type=int, default=100000, help="Radius (bp) for the nearby-call count feature")
    sv_parser.add_argument("--window", type=int, default=50, help="Sequence-context half-window (bp) at each breakpoint")
    sv_parser.add_argument("--threads", type=int, default=8)
    sv_parser.add_argument("--hash-input", action="store_true", help="Record input SHA-256 in metadata")
    _add_model_root(sv_parser)

    inspect_parser = sub.add_parser("inspect", help="Show model registry and environment")
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")
    inspect_parser.add_argument("--checksum", action="store_true")
    _add_model_root(inspect_parser)

    test_parser = sub.add_parser("self-test", help="Verify artifacts and frozen predictions")
    test_parser.add_argument("--component", choices=["all", "str", "sv"], default="all")
    test_parser.add_argument("--checksum", action="store_true", help="Hash all selected weights (slow)")
    _add_model_root(test_parser)
    return main


def inspect_models(registry, root, checksum=False):
    result = {"tool_version": __version__, "model_root": str(root), "software": dependency_versions(), "models": {}}
    for name, entry in registry["models"].items():
        artifacts = {}
        for role in ROLES:
            if role not in entry:
                continue
            try:
                path = resolve_artifact(entry[role], root)
                observed_hash = verify_artifact(path, entry[role]) if checksum else None
                artifacts[role] = {"path": str(path), "exists": True, "bytes": path.stat().st_size, "sha256_verified": observed_hash is not None}
            except Exception as error:
                artifacts[role] = {"path": str(root / entry[role]["path"]), "exists": False, "error": str(error)}
        result["models"][name] = {"description": entry["description"], "artifacts": artifacts}
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    registry = load_registry()
    root = model_root(args.model_root)
    try:
        if args.command == "str":
            summary, metadata = score_str(args, registry, root)
            print(f"STR: {json.dumps(summary)}", file=sys.stderr)
            print(f"outputs -> {[p for p in (args.out_vcf, args.out_tsv) if p]}; metadata -> {metadata}", file=sys.stderr)
        elif args.command == "sv":
            summary, metadata = score_sv(args, registry, root)
            print(f"SV: {json.dumps(summary)}", file=sys.stderr)
            print(f"outputs -> {[p for p in (args.out_vcf, args.out_tsv) if p]}; metadata -> {metadata}", file=sys.stderr)
        elif args.command == "inspect":
            result = inspect_models(registry, root, args.checksum)
            if args.as_json:
                print(json.dumps(result, indent=2))
            else:
                print(f"annstrata {result['tool_version']} | model root: {result['model_root']}")
                for name, item in result["models"].items():
                    status = "OK" if all(x.get("exists") for x in item["artifacts"].values()) else "MISSING"
                    print(f"  {name:<6} {status}  {item['description']}")
        elif args.command == "self-test":
            run_selftest(registry, root, args.component, args.checksum)
    except (FileNotFoundError, RuntimeError, ValueError, AssertionError) as error:
        print(f"annstrata: error: {error}", file=sys.stderr)
        return 2
    return 0
