"""Rank saved volume-model comparisons without running any training."""

import argparse
from pathlib import Path

from compare_volume_architectures import save_architecture_rankings

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summaries",
        nargs="+",
        type=Path,
        help="architecture_summary.json files from matching-split experiments",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "volume_results" / "architecture_rankings.json",
    )
    args = parser.parse_args()
    rankings = save_architecture_rankings(args.summaries, args.output)
    for profile, rows in rankings.items():
        print(f"\n{profile.replace('_', ' ').title()} priority:")
        for position, row in enumerate(rows, start=1):
            print(
                f"{position}. {row['architecture']}: regret="
                f"{row['weighted_relative_regret']:.4f}; "
                f"whole={row['mean_design_rmse_pa']:.3f} Pa; "
                f"near-body={row['mean_near_wall_rmse_pa']:.3f} Pa; "
                f"worst={row['worst_design_rmse_pa']:.3f} Pa"
            )
    print(f"\nSaved ranking report: {args.output}")


if __name__ == "__main__":
    main()