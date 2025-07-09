"""
segformer_predictor.py

A tiny OOP wrapper around NVIDIA’s SegFormer B5 checkpoint that
illustrates exactly the same workflow as the original notebook but
packs it into a reusable class.

Dependencies (same as before):
  pip install transformers safetensors huggingface_hub pillow matplotlib
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


class SegFormerPredictor:
    """Minimal, single-file convenience wrapper for SegFormer inference."""

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        device: Union[str, torch.device, None] = None,
    ):
        self.model_name = model_name
        self.device = (
            torch.device(device)
            if isinstance(device, str)
            else device
            if isinstance(device, torch.device)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # keep the original “no resize” behaviour
        self.processor = SegformerImageProcessor(do_resize=False)
        self.model: SegformerForSemanticSegmentation | None = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def load_model(self, use_safetensors: bool = True) -> None:
        """Downloads and places the SegFormer checkpoint on the chosen device."""
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(
                self.model_name, use_safetensors=use_safetensors
            )
            .to(self.device)
            .eval()
        )

    def infer_image(self, image: Image.Image) -> np.ndarray:
        """
        Runs a forward pass and returns a (H, W) numpy array of class IDs
        already resized to the input image’s resolution.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before infer_image().")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(pixel_values)

        seg_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        return seg_map.cpu().numpy()

    # ------------------------------------------------------------------
    # helper (optional): fast ADE20K palette for pretty colour maps
    # ------------------------------------------------------------------
    @staticmethod
    def ade_palette() -> list[list[int]]:
        """ADE20K palette that maps each class to RGB values."""
        # (identical list as in the original notebook, truncated for brevity)
        return [
            [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]
        ]


# ----------------------------------------------------------------------
# demo usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    predictor = SegFormerPredictor()
    predictor.load_model()  # weights are cached after the first run

    # ------------------------------------------------------------------
    # download a sample image (ADE20K validation sample #1)
    # ------------------------------------------------------------------
    repo = "hf-internal-testing/fixtures_ade20k"
    img_path = hf_hub_download(repo_id=repo, filename="ADE_val_00000001.jpg", repo_type="dataset")
    image = Image.open(img_path)

    # ------------------------------------------------------------------
    # run inference
    # ------------------------------------------------------------------
    seg = predictor.infer_image(image)

    # ------------------------------------------------------------------
    # colourise the prediction for visual inspection
    # ------------------------------------------------------------------
    palette = np.array(predictor.ade_palette(), dtype=np.uint8)
    colour = palette[seg]  # (H, W, 3)
    blend = (0.5 * np.asarray(image) + 0.5 * colour[..., ::-1]).astype(np.uint8)

    plt.figure(figsize=(12, 8))
    plt.imshow(blend)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
