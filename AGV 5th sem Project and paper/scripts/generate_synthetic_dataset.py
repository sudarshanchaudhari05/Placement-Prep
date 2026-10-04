"""
Synthetic Dataset Generation Command-Line Tool
==============================================
CLI interface supporting:
- Small validation run: python scripts/generate_synthetic_dataset.py --num-samples 20
- Full research run:    python scripts/generate_synthetic_dataset.py --num-samples 6000 --output-dir data
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.generator import SyntheticDatasetGenerator
from scripts.generate_qc_artifacts import (
    generate_file_manifest,
    generate_qa_and_statistics_report,
    generate_all_contact_sheets,
)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic AGV path tracking dataset")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6000,
        help="Number of samples to generate (default: 6000 for full research dataset)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1000,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Root output directory",
    )
    parser.add_argument(
        "--generate-previews",
        action="store_true",
        default=True,
        help="Generate visual quality control contact sheets",
    )

    args = parser.parse_args()

    out_p = Path(args.output_dir)
    print(f"================================================================================")
    print(f"STARTING RESEARCH DATASET GENERATION: TARGET {args.num_samples} FRAMES")
    print(f"Output directory: {out_p.resolve()}")
    print(f"Base seed: {args.base_seed}")
    print(f"================================================================================")

    generator = SyntheticDatasetGenerator(
        base_dir=out_p,
        base_seed=args.base_seed,
    )

    stats = generator.generate_dataset(num_samples=args.num_samples, progress_interval=500)
    print(f"\nDataset generation completed successfully: {stats}")

    print("\nRunning comprehensive Quality Assurance (QA) and integrity audit...")
    qa_report = generate_qa_and_statistics_report(data_dir=out_p)

    print("\nGenerating SHA-256 integrity manifest for all generated images...")
    manifest = generate_file_manifest(data_dir=out_p)

    if args.generate_previews:
        print("\nRendering all 6 visual quality control contact sheets...")
        generate_all_contact_sheets(data_dir=out_p)

    print(f"\n================================================================================")
    print(f"STEP 4 COMPLETE: FULL RESEARCH DATASET READY ({stats['total_generated']} FRAMES)")
    print(f"================================================================================")


if __name__ == "__main__":
    main()
