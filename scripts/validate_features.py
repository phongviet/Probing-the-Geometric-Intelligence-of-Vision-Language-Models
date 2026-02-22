import numpy as np
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Directory not found: {output_dir}")
        sys.exit(1)

    files = list(output_dir.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {output_dir}")
        return

    print(f"Found {len(files)} files in {output_dir}")

    # Check first file for detailed inspection
    first_file = files[0]
    try:
        data = np.load(first_file)
        print(f"Inspecting {first_file.name}:")
        if "global_features" not in data or "local_features" not in data:
            print("  ERROR: Missing keys 'global_features' or 'local_features'")
        else:
            g_feat = data["global_features"]
            l_feat = data["local_features"]
            print(f"  Global shape: {g_feat.shape} (Expected: [20, D])")
            print(f"  Local shape: {l_feat.shape} (Expected: [20, N, D])")

            if g_feat.shape[0] != 20:
                print(
                    f"  WARNING: Global features batch dim is {g_feat.shape[0]}, expected 20"
                )
            if l_feat.shape[0] != 20:
                print(
                    f"  WARNING: Local features batch dim is {l_feat.shape[0]}, expected 20"
                )

    except Exception as e:
        print(f"Error reading {first_file}: {e}")


if __name__ == "__main__":
    main()
