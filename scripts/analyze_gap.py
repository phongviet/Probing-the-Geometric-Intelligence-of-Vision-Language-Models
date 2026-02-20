"""
scripts/analyze_gap.py
----------------------
Analyze the "Geometric Gap" between Linear and MLP probes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="experiments/results.json")
    parser.add_argument("--output_dir", type=str, default="experiments/analysis")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    if df.empty:
        print("No results found.")
        return

    # Filter for relevant columns
    # Metrics differ by task

    # Calculate Geometric Gap
    # Pivot table: Index=[task, backbone, layer], Columns=[probe], Values=[metric]

    tasks = df["task"].unique()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_tables = []

    for task in tasks:
        print(f"Analyzing Task: {task}")
        task_df = df[df["task"] == task].copy()

        if task == "rotation":
            metric = "accuracy"
        elif task == "symmetry":
            metric = "f1_macro"  # Or subset_accuracy
        elif task == "normals":
            metric = "mae_degrees"
        else:
            continue

        # Pivot to get Linear and MLP side-by-side
        pivot = task_df.pivot_table(
            index=["backbone", "layer"], columns="probe", values=metric
        ).reset_index()

        if "linear" in pivot.columns and "mlp" in pivot.columns:
            if task == "normals":
                # For error, Gap = Linear - MLP (Positive means MLP is better/lower error)
                # Wait, Gap definition: Perf_MLP - Perf_Linear.
                # If metric is Error, lower is better.
                # So if MLP is better (lower error), Gap should be positive?
                # Usually Gap = Perf_NonLinear - Perf_Linear.
                # If Perf is Accuracy, Gap > 0 means NonLinear > Linear.
                # If Perf is Error, we want Gap > 0 to mean NonLinear is better.
                # So Gap = Linear_Error - MLP_Error.
                pivot["geometric_gap"] = pivot["linear"] - pivot["mlp"]
            else:
                pivot["geometric_gap"] = pivot["mlp"] - pivot["linear"]

        print(pivot)
        pivot.to_csv(output_dir / f"{task}_gap_analysis.csv", index=False)
        summary_tables.append((task, pivot))

    # Generate LateX or Markdown summary
    summary_md = "# Geometric Gap Analysis\n\n"
    for task, table in summary_tables:
        summary_md += f"## {task.capitalize()}\n"
        summary_md += table.to_markdown(index=False)
        summary_md += "\n\n"

    with open(output_dir / "summary.md", "w") as f:
        f.write(summary_md)

    print(f"Analysis saved to {output_dir}")


if __name__ == "__main__":
    main()
