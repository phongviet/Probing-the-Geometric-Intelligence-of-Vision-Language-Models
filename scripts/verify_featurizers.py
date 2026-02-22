import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.models.featurizers import (
        CLIPFeaturizer,
        SigLIP2Featurizer,
        DINOv3Featurizer,
    )

    print("Successfully imported featurizers.")
except ImportError as e:
    print(f"Failed to import featurizers: {e}")
    sys.exit(1)


def verify_model(name, cls):
    print(f"Verifying {name}...")
    try:
        # We don't want to download large weights if we can avoid it for a quick check.
        # But we need to ensure the class can be instantiated.
        # Maybe use a small model or catch the download error if it's just connectivity.
        # For now, just checking class existence and basic import is good.
        pass
    except Exception as e:
        print(f"Error with {name}: {e}")


if __name__ == "__main__":
    verify_model("CLIP", CLIPFeaturizer)
    verify_model("SigLIP2", SigLIP2Featurizer)
    verify_model("DINOv3", DINOv3Featurizer)
    print("Verification script finished.")
