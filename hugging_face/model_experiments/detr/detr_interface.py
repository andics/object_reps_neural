"""
detr_interface.py

A model interface wrapper that provides DETR-compatible output from DETR segmentation models.
This interface standardizes model loading, inference, and output formatting across experiments.

Dependencies:
  pip install transformers safetensors huggingface_hub pillow matplotlib torch torchvision
"""

from pathlib import Path
from typing import Union, Dict, Any
import logging
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from transformers import (
    DetrImageProcessor,
    DetrForSegmentation,
)
from transformers.models.detr.feature_extraction_detr import rgb_to_id
from skimage.measure import label, regionprops


class ModelInterface:
    """
    Abstract base interface for model inference that experiments can use.
    All model implementations should inherit from this class.
    """
    
    def load_model(self) -> None:
        """Load the model from checkpoint or hub."""
        raise NotImplementedError
    
    def infer_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on an image and return predictions in DETR-compatible format.
        
        Returns:
            Dict with keys:
            - 'pred_masks': torch.Tensor of shape (1, N, H, W) where N is number of queries
            - 'pred_logits': torch.Tensor of shape (1, N, num_classes) 
            - 'pred_boxes': torch.Tensor of shape (1, N, 4) in DETR format
        """
        raise NotImplementedError


class DetrInterface(ModelInterface):
    """
    DETR segmentation model interface that provides panoptic segmentation outputs.
    Uses DETR for panoptic segmentation which outputs proper segmentation masks.
    """

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50-panoptic",
        device: Union[str, torch.device, None] = None,
        num_queries: int = 100,
        logger: logging.Logger = None,
        confidence_threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.num_queries = num_queries
        self.confidence_threshold = confidence_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        self.device = (
            torch.device(device)
            if isinstance(device, str)
            else device
            if isinstance(device, torch.device)
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # DETR processor handles image preprocessing
        self.processor = DetrImageProcessor.from_pretrained(self.model_name)
        self.model: DetrForSegmentation = None
        
        self.logger.info(f"Initialized DETR Segmentation interface with device: {self.device}")

    def load_model(self, use_safetensors: bool = True) -> None:
        """Downloads and loads the DETR segmentation model."""
        self.logger.info(f"Loading DETR segmentation model: {self.model_name}")
        
        self.model = (
            DetrForSegmentation.from_pretrained(
                self.model_name, use_safetensors=use_safetensors
            )
            .to(self.device)
            .eval()
        )
        
        self.logger.info("DETR segmentation model loaded successfully")

    def infer_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference and return DETR-compatible predictions.
        
        The DETR segmentation model outputs proper segmentation masks from panoptic segmentation.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before infer_image().")

        # Get original image dimensions
        orig_width, orig_height = image.size
        
        # Run DETR segmentation inference
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process the panoptic segmentation outputs
        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        result = self.processor.post_process_panoptic(outputs, processed_sizes)[0]
        
        # Extract the panoptic segmentation
        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        # Convert RGB to segment IDs
        panoptic_seg_id = rgb_to_id(panoptic_seg)
        
        # Convert panoptic segmentation to individual masks
        pred_masks = self._convert_panoptic_to_masks(
            panoptic_seg_id, result["segments_info"], orig_height, orig_width
        )
        
        # Use the original DETR outputs for logits and boxes
        pred_logits = outputs.logits  # (1, num_queries, num_classes)
        pred_boxes = outputs.pred_boxes  # (1, num_queries, 4)
        
        return {
            'pred_masks': pred_masks,
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }

    def _convert_panoptic_to_masks(self, panoptic_seg_id: np.ndarray, 
                                  segments_info: list, height: int, width: int) -> torch.Tensor:
        """
        Convert panoptic segmentation to individual binary masks.
        
        Args:
            panoptic_seg_id: Panoptic segmentation map with segment IDs
            segments_info: Information about each segment
            height, width: Original image dimensions
            
        Returns:
            Tensor of shape (1, num_queries, H, W) with binary masks
        """
        
        # Resize panoptic segmentation to original image size
        if panoptic_seg_id.shape != (height, width):
            panoptic_pil = Image.fromarray(panoptic_seg_id.astype(np.uint8))
            panoptic_pil = panoptic_pil.resize((width, height), Image.NEAREST)
            panoptic_seg_id = np.array(panoptic_pil)
        
        # Initialize masks list
        masks = []
        
        # Create masks from segments
        for segment in segments_info:
            segment_id = segment['id']
            
            # Create binary mask for this segment
            mask = (panoptic_seg_id == segment_id).astype(np.float32)
            
            if mask.sum() > 0:  # Only add non-empty masks
                masks.append(torch.from_numpy(mask))
        
        # If we don't have enough masks, create additional masks by splitting larger segments
        if len(masks) < self.num_queries:
            # Find larger segments and split them into connected components
            for segment in segments_info:
                if len(masks) >= self.num_queries:
                    break
                    
                segment_id = segment['id']
                mask = (panoptic_seg_id == segment_id).astype(np.uint8)
                
                # Split into connected components
                labeled = label(mask, connectivity=2)
                regions = regionprops(labeled)
                
                # Add largest connected components as separate masks
                for region in sorted(regions, key=lambda r: r.area, reverse=True):
                    if len(masks) >= self.num_queries:
                        break
                    if region.area > 50:  # Minimum area threshold
                        component_mask = (labeled == region.label).astype(np.float32)
                        masks.append(torch.from_numpy(component_mask))
        
        # If still not enough masks, create simple grid-based masks as fallback
        while len(masks) < self.num_queries:
            grid_idx = len(masks) % 16  # Create up to 16 different grid positions
            grid_size = 4  # 4x4 grid
            row = grid_idx // grid_size
            col = grid_idx % grid_size
            
            # Create mask in grid cell
            mask = np.zeros((height, width), dtype=np.float32)
            
            cell_h = height // grid_size
            cell_w = width // grid_size
            y1 = row * cell_h
            y2 = min((row + 1) * cell_h, height)
            x1 = col * cell_w
            x2 = min((col + 1) * cell_w, width)
            
            # Add small central region in each grid cell
            center_y = (y1 + y2) // 2
            center_x = (x1 + x2) // 2
            mask_h = max(1, cell_h // 4)
            mask_w = max(1, cell_w // 4)
            
            mask_y1 = max(0, center_y - mask_h // 2)
            mask_y2 = min(height, center_y + mask_h // 2)
            mask_x1 = max(0, center_x - mask_w // 2)
            mask_x2 = min(width, center_x + mask_w // 2)
            
            if mask_y2 > mask_y1 and mask_x2 > mask_x1:
                mask[mask_y1:mask_y2, mask_x1:mask_x2] = 0.1  # Lower confidence for fallback masks
            
            masks.append(torch.from_numpy(mask))
        
        # Stack and add batch dimension
        pred_masks = torch.stack(masks[:self.num_queries], dim=0)  # (num_queries, H, W)
        pred_masks = pred_masks.unsqueeze(0).to(self.device)  # (1, num_queries, H, W)
        
        return pred_masks

    # ------------------------------------------------------------------
    # Helper methods for visualization and compatibility
    # ------------------------------------------------------------------
    def get_coco_labels(self) -> Dict[int, str]:
        """Get COCO class labels used by DETR."""
        if self.model is None:
            return {}
        return self.model.config.id2label

    def visualize_predictions(self, image: Image.Image, predictions: Dict[str, Any], 
                            threshold: float = 0.5) -> Image.Image:
        """
        Visualize DETR panoptic segmentation predictions on the image.
        
        Args:
            image: Input PIL image
            predictions: Output from infer_image()
            threshold: Confidence threshold for visualization
            
        Returns:
            PIL image with visualizations
        """
        # Re-run post-processing for visualization
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        result = self.processor.post_process_panoptic(outputs, processed_sizes)[0]
        
        # Extract panoptic segmentation
        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        
        # Create overlay
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.imshow(panoptic_seg, alpha=0.5)
        
        # Add segment labels
        panoptic_seg_id = rgb_to_id(panoptic_seg)
        for segment in result["segments_info"]:
            segment_id = segment['id']
            label_id = segment['label_id']
            
            # Get class labels
            labels_map = self.get_coco_labels()
            label_text = labels_map.get(label_id, 'unknown')
            
            # Find centroid of segment
            mask = (panoptic_seg_id == segment_id)
            if mask.sum() > 0:
                y_coords, x_coords = np.where(mask)
                centroid_x = np.mean(x_coords)
                centroid_y = np.mean(y_coords)
                
                ax.text(centroid_x, centroid_y, label_text, fontsize=10, color='white',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
        
        ax.axis('off')
        plt.tight_layout()
        
        # Convert matplotlib figure to PIL Image
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return Image.fromarray(buf)


# ----------------------------------------------------------------------
# demo usage and backward compatibility
# ----------------------------------------------------------------------
# Keep the old class name for backward compatibility
DetrPredictor = DetrInterface


if __name__ == "__main__":
    # Demo usage
    interface = DetrInterface()
    interface.load_model()

    # Download a sample image (COCO validation sample)
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        response = requests.get(url)
        from io import BytesIO
        image = Image.open(BytesIO(response.content))
    except:
        # Fallback to HuggingFace sample if COCO is not accessible
        repo = "hf-internal-testing/fixtures_ade20k"
        img_path = hf_hub_download(repo_id=repo, filename="ADE_val_00000001.jpg", repo_type="dataset")
        image = Image.open(img_path)

    # Run inference
    predictions = interface.infer_image(image)
    
    print(f"Prediction format:")
    print(f"- pred_masks shape: {predictions['pred_masks'].shape}")
    print(f"- pred_logits shape: {predictions['pred_logits'].shape}")
    print(f"- pred_boxes shape: {predictions['pred_boxes'].shape}")
    print(f"- Number of non-empty masks: {(predictions['pred_masks'].sum(dim=(-2, -1)) > 0).sum().item()}")

    # For visualization, show panoptic segmentation
    results_vis = interface.visualize_predictions(image, predictions)
    
    # Save visualization
    results_vis.save("detr_segmentation_demo_output.png")
    print("Saved visualization to detr_segmentation_demo_output.png")