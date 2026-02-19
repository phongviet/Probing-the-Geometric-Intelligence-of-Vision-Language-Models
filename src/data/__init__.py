"""
src/data/__init__.py
Exports all three GIQ dataset classes.
"""

from .mental_rotation import MentalRotationDataset
from .symmetry import SymmetryDataset
from .normals import NormalsDataset

__all__ = ["MentalRotationDataset", "SymmetryDataset", "NormalsDataset"]
