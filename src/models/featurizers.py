from __future__ import annotations

import abc
import warnings
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    AutoImageProcessor,
)


class AbstractFeaturizer(nn.Module, abc.ABC):
    """Abstract base class for all vision featurizers."""

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device

    @abc.abstractmethod
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extract features from images.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images, typically [B, C, H, W] and normalized.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - "global": [B, D] global feature vector (CLS token or pooled)
            - "local": [B, N, D] local feature patches
        """
        pass

    @abc.abstractmethod
    def get_transform(self) -> Any:
        """Return the torchvision transform or processor required by this model."""
        pass


class CLIPFeaturizer(AbstractFeaturizer):
    """
    CLIP ViT-B/16 featurizer.
    Uses generic CLIPModel from transformers.
    """

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch16", device: str = "cuda"
    ):
        super().__init__(model_name, device)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model.vision_model(
                pixel_values=images, output_hidden_states=True
            )
            last_hidden = outputs.last_hidden_state
            cls_token = last_hidden[:, 0, :]
            patch_tokens = last_hidden[:, 1:, :]

            return {"global": cls_token, "local": patch_tokens}

    def get_transform(self) -> Any:
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )


class SigLIP2Featurizer(AbstractFeaturizer):
    """
    SigLIP 2 featurizer.
    Using google/siglip2 checkpoints.
    """

    def __init__(
        self, model_name: str = "google/siglip2-base-patch16-224", device: str = "cuda"
    ):
        super().__init__(model_name, device)
        try:
            self.model = AutoModel.from_pretrained(model_name).to(device)
        except OSError:
            warnings.warn(
                f"Model {model_name} not found. Ensure you have access or connection."
            )
            raise

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            if hasattr(self.model, "vision_model"):
                outputs = self.model.vision_model(pixel_values=images)
            else:
                outputs = self.model(pixel_values=images)

            last_hidden = outputs.last_hidden_state  # [B, N, D]

            # Heuristic for CLS token presence
            seq_len = last_hidden.shape[1]
            if seq_len == 197:  # likely CLS + 196 patches
                return {"global": last_hidden[:, 0, :], "local": last_hidden[:, 1:, :]}
            else:
                # Mean pool for global
                return {"global": last_hidden.mean(dim=1), "local": last_hidden}

    def get_transform(self) -> Any:
        try:
            image_mean = self.processor.image_processor.image_mean
            image_std = self.processor.image_processor.image_std
            size = self.processor.image_processor.size["height"]
        except (AttributeError, KeyError):
            image_mean = (0.5, 0.5, 0.5)
            image_std = (0.5, 0.5, 0.5)
            size = 224

        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std),
            ]
        )


class DINOv3Featurizer(AbstractFeaturizer):
    """
    DINOv3 featurizer.
    Using facebook/dinov3 checkpoints.
    """

    def __init__(self, model_name: str = "facebook/dinov3-base", device: str = "cuda"):
        super().__init__(model_name, device)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.eval()

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(pixel_values=images)
            last_hidden = outputs.last_hidden_state

            if last_hidden.shape[1] > 196:  # Assuming patch 16, 224 size
                cls_token = last_hidden[:, 0, :]
                patch_tokens = last_hidden[:, 1:, :]
            else:
                cls_token = last_hidden.mean(dim=1)
                patch_tokens = last_hidden

            return {"global": cls_token, "local": patch_tokens}

    def get_transform(self) -> Any:
        from torchvision import transforms

        try:
            image_mean = self.processor.image_mean
            image_std = self.processor.image_std
            size = self.processor.size["height"]
        except (AttributeError, KeyError):
            image_mean = (0.485, 0.456, 0.406)
            image_std = (0.229, 0.224, 0.225)
            size = 224

        return transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std),
            ]
        )
