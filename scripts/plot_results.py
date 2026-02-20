"""
scripts/plot_results.py
-----------------------
Generate final charts and plots for the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", type=str, default="experiments/analysis")
    parser.add_argument("--output_dir", type=str, default="experiments/plots")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(analysis_dir.glob("*_gap_analysis.csv"))
    if not csv_files:
        print(f"No analysis files found in {analysis_dir}")
        return

    # Set style manually since seaborn is not available
    plt.style.use("ggplot")

    for csv_file in csv_files:
        task_name = csv_file.stem.split("_")[0]
        print(f"Plotting for task: {task_name}")

        df = pd.read_csv(csv_file)

        # Determine number of groups (backbones) and bar width
        if "backbone" not in df.columns:
            print(f"Skipping {csv_file}: 'backbone' column missing.")
            continue

        backbones = df["backbone"].unique()
        n_backbones = len(backbones)
        x = range(n_backbones)
        width = 0.35

        # Determine unique layers if present, else default to dummy
        if "layer" in df.columns:
            layers = df["layer"].unique()
        else:
            layers = ["default"]
            df["layer"] = "default"

        for layer in layers:
            # Filter for this layer
            layer_df = df[df["layer"] == layer].copy()

            # Ensure we have all backbones represented (fill missing with 0)
            # Create a full index dataframe
            full_df = pd.DataFrame({"backbone": backbones})
            layer_df = pd.merge(full_df, layer_df, on="backbone", how="left")

            # Plot 1: Performance (Linear vs MLP)
            plt.figure(figsize=(10, 6))

            # Handle potential NaNs if a backbone is missing for a layer
            lin_data = (
                layer_df["linear"].fillna(0)
                if "linear" in layer_df.columns
                else [0] * n_backbones
            )
            mlp_data = (
                layer_df["mlp"].fillna(0)
                if "mlp" in layer_df.columns
                else [0] * n_backbones
            )

            # Positions
            x_lin = [i - width / 2 for i in x]
            x_mlp = [i + width / 2 for i in x]

            rects1 = plt.bar(x_lin, lin_data, width, label="Linear")
            rects2 = plt.bar(x_mlp, mlp_data, width, label="MLP")

            title_suffix = f" ({layer})" if layer != "default" else ""
            plt.title(f"{task_name.capitalize()} Performance{title_suffix}")
            plt.ylabel("Score" if task_name != "normals" else "Error (Lower is Better)")
            plt.xlabel("Backbone")
            plt.xticks(x, backbones)
            plt.legend()

            # Add labels
            plt.bar_label(rects1, fmt="%.2f", padding=3)
            plt.bar_label(rects2, fmt="%.2f", padding=3)

            plt.tight_layout()
            save_name = (
                f"{task_name}_{layer}_performance.png"
                if layer != "default"
                else f"{task_name}_performance.png"
            )
            plt.savefig(output_dir / save_name)
            plt.close()

            # Plot 2: Geometric Gap
            if "geometric_gap" in layer_df.columns:
                plt.figure(figsize=(8, 6))
                gap_data = layer_df["geometric_gap"].fillna(0)
                colors = ["green" if v > 0 else "red" for v in gap_data]

                rects_gap = plt.bar(x, gap_data, color=colors)

                plt.title(f"{task_name.capitalize()} Geometric Gap{title_suffix}")
                plt.ylabel("Gap (Positive = Non-Linear Advantage)")
                plt.axhline(0, color="black", linestyle="--", linewidth=1)
                plt.xticks(x, backbones)

                plt.bar_label(rects_gap, fmt="%.2f", padding=3)

                plt.tight_layout()
                save_name_gap = (
                    f"{task_name}_{layer}_gap.png"
                    if layer != "default"
                    else f"{task_name}_gap.png"
                )
                plt.savefig(output_dir / save_name_gap)
                plt.close()

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
