#!/usr/bin/env python3
"""Detect implementation drift in the simulator-v3 methods manuscript.

The hash is deliberately an acknowledgement gate, not a documentation
generator.  Review and update the manuscript first; then refresh its marker
with ``--update --acknowledge-review``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPOSITORY_ROOT / "repo_reports" / "METHODS_MANUSCRIPT.md"
SOURCE_FILES = (
    "shared_job_scratch.sh",
    "arch/config.py",
    "arch/utils.py",
    "arch/data.py",
    "arch/networks.py",
    "arch/circular_spline.py",
    "arch/train.py",
    "arch/train_model.py",
    "arch/model_registry.py",
    "arch/tf_prior.py",
    "arch/cache_contract.py",
    "arch/cache_posteriors.py",
    "arch/diagnostics/shear_bias_report.py",
    "arch/pretrain_ccl.slurm",
    "arch/train_npe.slurm",
    "arch/cache_posteriors.slurm",
    "arch/diagnostics/shear_bias_report.slurm",
    "data_generate/latin_hypercube.py",
    "data_generate/desi_test_set_sampling.py",
    "data_generate/generate_desi_test_sets.slurm",
    "data_generate/observation_schema.py",
    "data_generate/generate_fits.py",
    "data_generate/generate_fits_wrapper.py",
    "data_generate/generation_integrity.py",
    "data_generate/make_database.py",
    "data_generate/generate_simulator_v3.slurm",
    "data_generate/make_database_simulator_v3.slurm",
    "data_generate/merge_database_simulator_v3.slurm",
    "Accelerating_Kinematic_Lensing_Inference_with_Neural_Networks/main.tex",
    "Accelerating_Kinematic_Lensing_Inference_with_Neural_Networks/sections/data-generate.tex",
    "Accelerating_Kinematic_Lensing_Inference_with_Neural_Networks/sections/desi-kl.tex",
    "Accelerating_Kinematic_Lensing_Inference_with_Neural_Networks/sections/kl-basics.tex",
    "Accelerating_Kinematic_Lensing_Inference_with_Neural_Networks/sections/nn-arch.tex",
)
MARKER = re.compile(
    r"<!-- klnn-methods-source-sha256: ([0-9a-f]{64}|PENDING) -->"
)


def source_fingerprint() -> str:
    """Hash monitored paths and contents in a deterministic order."""

    digest = hashlib.sha256()
    for relative_path in SOURCE_FILES:
        source = REPOSITORY_ROOT / relative_path
        if not source.is_file():
            raise FileNotFoundError(f"monitored methods source is missing: {source}")
        encoded_path = relative_path.encode("utf-8")
        contents = source.read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def documented_fingerprint(text: str | None = None) -> str:
    """Return the unique fingerprint recorded in the manuscript."""

    manuscript_text = MANUSCRIPT.read_text(encoding="utf-8") if text is None else text
    matches = MARKER.findall(manuscript_text)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one methods fingerprint marker in {MANUSCRIPT}; "
            f"found {len(matches)}"
        )
    return matches[0]


def check() -> tuple[bool, str, str]:
    """Return whether the manuscript marker matches the monitored sources."""

    expected = source_fingerprint()
    recorded = documented_fingerprint()
    return recorded == expected, recorded, expected


def update_marker(*, acknowledge_review: bool) -> str:
    """Refresh the marker only after an explicit review acknowledgement."""

    if not acknowledge_review:
        raise ValueError(
            "refusing to update the fingerprint without --acknowledge-review; "
            "first reconcile every affected claim in METHODS_MANUSCRIPT.md"
        )
    text = MANUSCRIPT.read_text(encoding="utf-8")
    documented_fingerprint(text)
    fingerprint = source_fingerprint()
    updated, count = MARKER.subn(
        f"<!-- klnn-methods-source-sha256: {fingerprint} -->", text
    )
    if count != 1:
        raise RuntimeError("methods fingerprint replacement was not unique")
    MANUSCRIPT.write_text(updated, encoding="utf-8")
    return fingerprint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true", help="check for drift (default)")
    actions.add_argument("--show", action="store_true", help="print the current source hash")
    actions.add_argument("--update", action="store_true", help="write the current hash")
    parser.add_argument(
        "--acknowledge-review",
        action="store_true",
        help="confirm that affected manuscript claims were reviewed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.update:
            print(update_marker(acknowledge_review=args.acknowledge_review))
            return 0
        if args.acknowledge_review:
            raise ValueError("--acknowledge-review is valid only with --update")
        if args.show:
            print(source_fingerprint())
            return 0
        matches, recorded, expected = check()
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"methods-manuscript check failed: {error}", file=sys.stderr)
        return 2
    if matches:
        print(f"methods manuscript is synchronized: {expected}")
        return 0
    print(
        "METHODS_MANUSCRIPT.md is stale.\n"
        f"  recorded: {recorded}\n"
        f"  current:  {expected}\n"
        "Review and update the affected prose, then run:\n"
        "  python repo_reports/check_methods_manuscript.py "
        "--update --acknowledge-review",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
