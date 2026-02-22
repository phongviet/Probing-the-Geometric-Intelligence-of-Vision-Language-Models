"""
Download the GIQ 3D meshes (22.8 MB) from Google Drive.

Usage:
    conda run -n geoprobe python scripts/download_giq_meshes.py

The rendered images (90 GB) must be downloaded manually from:
    https://drive.google.com/file/d/1kCXKpisGIcz7qgKpdMFqzNji7CnNIX9w/view?usp=sharing
and extracted to:
    data/giq/renderings/
"""

import zipfile
from pathlib import Path

# Install gdown if needed: pip install gdown
try:
    import gdown
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
    import gdown

ROOT = Path(__file__).resolve().parent.parent
MESHES_DIR = ROOT / "data" / "giq" / "meshes"
MESHES_DIR.mkdir(parents=True, exist_ok=True)

MESHES_ID = "1i_6up_4Cc24EaIhnKkhMboaDw-1tdJdC"
ZIP_PATH = ROOT / "data" / "giq" / "3d_meshes.zip"

if not ZIP_PATH.exists():
    print(f"Downloading 3D meshes to {ZIP_PATH} ...")
    url = f"https://drive.google.com/uc?id={MESHES_ID}"
    gdown.download(url, str(ZIP_PATH), quiet=False)
else:
    print(f"Zip already exists at {ZIP_PATH}, skipping download.")

print(f"Extracting to {MESHES_DIR} ...")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(MESHES_DIR)

print("Done. Mesh files available at:", MESHES_DIR)

# Print summary
obj_files = list(MESHES_DIR.rglob("*.obj"))
print(f"Found {len(obj_files)} OBJ files.")
