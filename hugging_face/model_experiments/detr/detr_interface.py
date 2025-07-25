 """
detr_interface.py

A model interface wrapper that provides DETR-compatible output from DETR models.
This interface standardizes model loading, inference, and output formatting across experiments.

Dependencies:
  pip install transformers safetensors huggingface_hub pillow matplotlib torch torchvision
"""

from pathlib import Path
from typing import Union, Dict, Any
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
)
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
    DETR model interface that provides compatible output format.
    Converts DETR object detection outputs to mask-based format for compatibility
    with existing segmentation-based experiments.
    """

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
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
        self.model: DetrForObjectDetection = None
        
        self.logger.info(f"Initialized DETR interface with device: {self.device}")

    def load_model(self, use_safetensors: bool = True) -> None:
        """Downloads and loads the DETR model."""
        self.logger.info(f"Loading DETR model: {self.model_name}")
        
        self.model = (
            DetrForObjectDetection.from_pretrained(
                self.model_name, use_safetensors=use_safetensors
            )
            .to(self.device)
            .eval()
        )
        
        self.logger.info("DETR model loaded successfully")

    def infer_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference and return DETR-compatible predictions.
        
        The DETR object detection outputs are converted to mask-based format
        by creating binary masks from bounding boxes for compatibility with
        existing segmentation-based experiments.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before infer_image().")

        # Get original image dimensions
        orig_width, orig_height = image.size
        
        # Run DETR inference
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process the outputs to get final predictions
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]
        
        # Convert to mask-based format for compatibility
        pred_masks = self._convert_boxes_to_masks(
            results, orig_height, orig_width
        )
        
        # Use the original DETR outputs for logits and boxes
        pred_logits = outputs.logits  # (1, num_queries, num_classes)
        pred_boxes = outputs.pred_boxes  # (1, num_queries, 4)
        
        return {
            'pred_masks': pred_masks,
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }

    def _convert_boxes_to_masks(self, results: Dict[str, torch.Tensor], 
                               height: int, width: int) -> torch.Tensor:
        """
        Convert DETR bounding box predictions to binary masks.
        
        Args:
            results: Post-processed DETR results with 'boxes', 'scores', 'labels'
            height, width: Original image dimensions
            
        Returns:
            Tensor of shape (1, num_queries, H, W) with binary masks
        """
        
        # Initialize empty masks
        masks = []
        
        # Get detected boxes (already filtered by confidence threshold)
        boxes = results.get("boxes", torch.empty(0, 4))  # (N, 4) in xyxy format
        scores = results.get("scores", torch.empty(0))
        labels = results.get("labels", torch.empty(0))
        
        # Create masks from detected boxes
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if len(masks) >= self.num_queries:
                break
                
            # Create binary mask from bounding box
            mask = torch.zeros((height, width), dtype=torch.float32)
            
            # Convert box coordinates to integers (boxes are in xyxy format)
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Fill the bounding box region
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1.0
            
            masks.append(mask)
        
        # If we don't have enough detected objects, create some additional masks
        # using a simple grid-based approach for compatibility
        while len(masks) < self.num_queries:
            # Create simple grid-based masks as fallback
            grid_idx = len(masks) % 16  # Create up to 16 different grid positions
            grid_size = 4  # 4x4 grid
            row = grid_idx // grid_size
            col = grid_idx % grid_size
            
            # Create mask in grid cell
            mask = torch.zeros((height, width), dtype=torch.float32)
            
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
            
            masks.append(mask)
        
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
        Visualize DETR predictions on the image.
        
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
        
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]
        
        # Draw results on image
        import matplotlib.patches as patches
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Get class labels
        labels_map = self.get_coco_labels()
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = box.cpu().numpy()
            score = score.cpu().item()
            label = label.cpu().item()
            
            # Create rectangle patch
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label and score
            label_text = f"{labels_map.get(label, 'unknown')}: {score:.2f}"
            ax.text(x1, y1 - 10, label_text, fontsize=10, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
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

    # For visualization, show detected objects
    results_vis = interface.visualize_predictions(image, predictions, threshold=0.5)
    
    # Save visualization
    results_vis.save("detr_demo_output.png")
    print("Saved visualization to detr_demo_output.png")