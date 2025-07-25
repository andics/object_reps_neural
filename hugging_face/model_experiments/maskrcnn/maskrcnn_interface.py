"""
maskrcnn_interface.py

A model interface wrapper that provides DETR-compatible output from Mask R-CNN models.
This interface standardizes model loading, inference, and output formatting across experiments.

Dependencies:
  pip install transformers safetensors huggingface_hub pillow matplotlib torch torchvision
"""

from pathlib import Path
from typing import Union, Dict, Any, List
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms.functional as TF
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


class MaskRCNNInterface(ModelInterface):
    """
    Mask R-CNN model interface that provides instance segmentation outputs.
    Uses torchvision's Mask R-CNN implementation which outputs proper segmentation masks.
    """

    def __init__(
        self,
        model_name: str = "maskrcnn_resnet50_fpn",
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
        
        # Define image preprocessing transforms
        self.transform = T.Compose([
            T.ToTensor(),
        ])
        
        self.model = None
        
        self.logger.info(f"Initialized Mask R-CNN interface with device: {self.device}")

    def load_model(self, pretrained: bool = True) -> None:
        """Downloads and loads the Mask R-CNN model."""
        self.logger.info(f"Loading Mask R-CNN model: {self.model_name}")
        
        if self.model_name == "maskrcnn_resnet50_fpn":
            self.model = maskrcnn_resnet50_fpn(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        self.model = self.model.to(self.device).eval()
        
        self.logger.info("Mask R-CNN model loaded successfully")

    def infer_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference and return DETR-compatible predictions.
        
        The Mask R-CNN model outputs proper instance segmentation masks.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before infer_image().")

        # Get original image dimensions
        orig_width, orig_height = image.size
        
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL to tensor
        img_tensor = TF.to_tensor(image).to(self.device)
        
        # Run Mask R-CNN inference
        with torch.no_grad():
            predictions = self.model([img_tensor])
        
        # Extract predictions for the first (and only) image
        pred = predictions[0]
        
        # Filter predictions by confidence threshold
        scores = pred['scores']
        keep = scores >= self.confidence_threshold
        
        filtered_boxes = pred['boxes'][keep]
        filtered_scores = scores[keep]
        filtered_labels = pred['labels'][keep]
        filtered_masks = pred['masks'][keep]
        
        # Convert to DETR-compatible format
        pred_masks = self._convert_masks_to_detr_format(
            filtered_masks, orig_height, orig_width
        )
        
        pred_logits = self._convert_labels_to_detr_format(
            filtered_labels, filtered_scores
        )
        
        pred_boxes = self._convert_boxes_to_detr_format(
            filtered_boxes, orig_height, orig_width
        )
        
        return {
            'pred_masks': pred_masks,
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }

    def _convert_masks_to_detr_format(self, masks: torch.Tensor, 
                                     height: int, width: int) -> torch.Tensor:
        """
        Convert Mask R-CNN masks to DETR format.
        
        Args:
            masks: Tensor of shape (N, 1, H, W) with instance masks
            height, width: Original image dimensions
            
        Returns:
            Tensor of shape (1, num_queries, H, W) with binary masks
        """
        if len(masks) == 0:
            # No detections, return empty masks
            empty_masks = torch.zeros(1, self.num_queries, height, width, 
                                    dtype=torch.float32, device=self.device)
            return empty_masks
        
        # Remove channel dimension and threshold masks
        masks = masks.squeeze(1)  # (N, H, W)
        binary_masks = (masks > 0.5).float()
        
        # Resize masks to original image size if needed
        if binary_masks.shape[-2:] != (height, width):
            binary_masks = F.interpolate(
                binary_masks.unsqueeze(1),  # Add channel dim for interpolation
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # Remove channel dim
            binary_masks = (binary_masks > 0.5).float()
        
        # Convert to list for easier manipulation
        mask_list = [binary_masks[i] for i in range(len(binary_masks))]
        
        # If we have more masks than num_queries, keep the top-scoring ones
        if len(mask_list) > self.num_queries:
            mask_list = mask_list[:self.num_queries]
        
        # If we don't have enough masks, create additional masks by splitting larger ones
        if len(mask_list) < self.num_queries:
            additional_masks = []
            for mask in mask_list:
                if len(mask_list) + len(additional_masks) >= self.num_queries:
                    break
                
                # Convert to numpy for connected components analysis
                mask_np = mask.cpu().numpy().astype(np.uint8)
                labeled = label(mask_np, connectivity=2)
                regions = regionprops(labeled)
                
                # Add largest connected components as separate masks
                for region in sorted(regions, key=lambda r: r.area, reverse=True):
                    if len(mask_list) + len(additional_masks) >= self.num_queries:
                        break
                    if region.area > 50:  # Minimum area threshold
                        component_mask = (labeled == region.label).astype(np.float32)
                        additional_masks.append(torch.from_numpy(component_mask).to(self.device))
            
            mask_list.extend(additional_masks)
        
        # If still not enough masks, create simple grid-based masks as fallback
        while len(mask_list) < self.num_queries:
            grid_idx = len(mask_list) % 16  # Create up to 16 different grid positions
            grid_size = 4  # 4x4 grid
            row = grid_idx // grid_size
            col = grid_idx % grid_size
            
            # Create mask in grid cell
            mask = torch.zeros((height, width), dtype=torch.float32, device=self.device)
            
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
            
            mask_list.append(mask)
        
        # Stack and add batch dimension
        pred_masks = torch.stack(mask_list[:self.num_queries], dim=0)  # (num_queries, H, W)
        pred_masks = pred_masks.unsqueeze(0)  # (1, num_queries, H, W)
        
        return pred_masks

    def _convert_labels_to_detr_format(self, labels: torch.Tensor, 
                                      scores: torch.Tensor) -> torch.Tensor:
        """
        Convert Mask R-CNN labels and scores to DETR format.
        
        Args:
            labels: Tensor of shape (N,) with class labels
            scores: Tensor of shape (N,) with confidence scores
            
        Returns:
            Tensor of shape (1, num_queries, num_classes) with logits
        """
        num_classes = 91  # COCO has 80 classes + background
        batch_size = 1
        
        # Initialize logits tensor
        pred_logits = torch.zeros(batch_size, self.num_queries, num_classes, 
                                 dtype=torch.float32, device=self.device)
        
        # Fill in logits for detected objects
        num_detections = min(len(labels), self.num_queries)
        for i in range(num_detections):
            label = labels[i].item()
            score = scores[i].item()
            
            # Convert score to logit (approximate)
            logit = torch.log(torch.tensor(score / (1 - score + 1e-8)))
            
            # Set logit for the predicted class
            if 0 <= label < num_classes:
                pred_logits[0, i, label] = logit
        
        # Set background class for empty queries
        for i in range(num_detections, self.num_queries):
            pred_logits[0, i, 0] = 5.0  # High confidence for background
        
        return pred_logits

    def _convert_boxes_to_detr_format(self, boxes: torch.Tensor, 
                                     height: int, width: int) -> torch.Tensor:
        """
        Convert Mask R-CNN boxes to DETR format.
        
        Args:
            boxes: Tensor of shape (N, 4) with boxes in [x1, y1, x2, y2] format
            height, width: Original image dimensions
            
        Returns:
            Tensor of shape (1, num_queries, 4) with boxes in DETR format [cx, cy, w, h]
        """
        batch_size = 1
        
        # Initialize boxes tensor
        pred_boxes = torch.zeros(batch_size, self.num_queries, 4, 
                               dtype=torch.float32, device=self.device)
        
        # Convert detected boxes
        num_detections = min(len(boxes), self.num_queries)
        for i in range(num_detections):
            x1, y1, x2, y2 = boxes[i]
            
            # Convert to center coordinates and normalize
            cx = (x1 + x2) / 2.0 / width
            cy = (y1 + y2) / 2.0 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            pred_boxes[0, i] = torch.tensor([cx, cy, w, h], device=self.device)
        
        return pred_boxes

    # ------------------------------------------------------------------
    # Helper methods for visualization and compatibility
    # ------------------------------------------------------------------
    def get_coco_labels(self) -> Dict[int, str]:
        """Get COCO class labels used by Mask R-CNN."""
        # COCO class names (index 0 is background)
        return {
            0: '__background__', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
            13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog',
            19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
            25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie',
            33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
            38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
            42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup',
            48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
            54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
            59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
            65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
            75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
            80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
            86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
        }

    def visualize_predictions(self, image: Image.Image, predictions: Dict[str, Any] = None, 
                            threshold: float = 0.5) -> Image.Image:
        """
        Visualize Mask R-CNN predictions on the image.
        
        Args:
            image: Input PIL image
            predictions: Output from infer_image() (optional, will recompute if None)
            threshold: Confidence threshold for visualization
            
        Returns:
            PIL image with visualizations
        """
        if predictions is None:
            predictions = self.infer_image(image)
        
        # Get the actual raw predictions for visualization
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = TF.to_tensor(image).to(self.device)
        
        with torch.no_grad():
            raw_predictions = self.model([img_tensor])
        
        pred = raw_predictions[0]
        
        # Filter by confidence
        scores = pred['scores']
        keep = scores >= threshold
        
        boxes = pred['boxes'][keep]
        labels = pred['labels'][keep]
        masks = pred['masks'][keep]
        scores = scores[keep]
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Get class labels
        labels_map = self.get_coco_labels()
        
        # Draw masks and boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
        
        for i, (box, label, mask, score, color) in enumerate(zip(boxes, labels, masks, scores, colors)):
            # Draw mask
            mask_np = mask.squeeze(0).cpu().numpy()
            mask_np = (mask_np > 0.5).astype(np.uint8)
            
            # Create colored mask
            colored_mask = np.zeros((*mask_np.shape, 4))
            colored_mask[:, :, :3] = color[:3]
            colored_mask[:, :, 3] = mask_np * 0.5  # Semi-transparent
            
            ax.imshow(colored_mask)
            
            # Draw bounding box
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                               fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            # Add label
            label_text = f"{labels_map.get(label.item(), 'unknown')}: {score:.2f}"
            ax.text(x1, y1 - 10, label_text, fontsize=10, color='white',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
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
MaskRCNNPredictor = MaskRCNNInterface


if __name__ == "__main__":
    # Demo usage
    interface = MaskRCNNInterface()
    interface.load_model()

    # Download a sample image (COCO validation sample)
    import requests
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    try:
        response = requests.get(url)
        from io import BytesIO
        image = Image.open(BytesIO(response.content))
    except:
        # Fallback to a simple test image
        image = Image.new('RGB', (640, 480), color='white')

    # Run inference
    predictions = interface.infer_image(image)
    
    print(f"Prediction format:")
    print(f"- pred_masks shape: {predictions['pred_masks'].shape}")
    print(f"- pred_logits shape: {predictions['pred_logits'].shape}")
    print(f"- pred_boxes shape: {predictions['pred_boxes'].shape}")
    print(f"- Number of non-empty masks: {(predictions['pred_masks'].sum(dim=(-2, -1)) > 0).sum().item()}")

    # For visualization, show instance segmentation
    results_vis = interface.visualize_predictions(image)
    
    # Save visualization
    results_vis.save("maskrcnn_segmentation_demo_output.png")
    print("Saved visualization to maskrcnn_segmentation_demo_output.png") 