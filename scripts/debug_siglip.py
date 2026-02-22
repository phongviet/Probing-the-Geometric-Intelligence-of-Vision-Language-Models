import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(".").resolve()))

from src.models.featurizers import SigLIP2Featurizer


def test_siglip2():
    print("Testing SigLIP2Featurizer...")
    try:
        featurizer = SigLIP2Featurizer(device="cpu")  # Use CPU for quick test
        print("Successfully initialized SigLIP2Featurizer")

        # Test forward pass with dummy image
        img = torch.randn(1, 3, 224, 224)
        out = featurizer(img)
        print("Forward pass successful")
        print("Global features shape:", out["global"].shape)
        print("Local features shape:", out["local"].shape)

    except Exception as e:
        print(f"Failed to initialize or run SigLIP2Featurizer: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_siglip2()
