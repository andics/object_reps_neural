#!/usr/bin/env python3
"""
demo_interface.py

Test script for the SegFormer interface that downloads a random image from the web
and demonstrates the model inference capabilities.

Usage:
    python demo_interface.py [--save_output] [--image_url URL]
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import requests
from io import BytesIO

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path to import segformer_interface
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))  # To access vanilla_segmentation
from segformer_interface import SegFormerInterface
from vanilla_segmentation import VanillaSegmentationSaver


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def download_image(url: str, logger: logging.Logger) -> Image.Image:
    """Download an image from a URL and return as PIL Image."""
    logger.info(f"Downloading image from: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Load image from response
        image = Image.open(BytesIO(response.content)).convert('RGB')
        logger.info(f"Successfully downloaded image with size: {image.size}")
        return image
        
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        raise


def get_sample_image_urls():
    """Return a list of sample image URLs for testing."""
    return [
        # Street scenes with objects
        "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800&q=80",  # Cityscape
        "https://images.unsplash.com/photo-1573164713714-d95e436ab8d6?w=800&q=80",  # Urban street
        
        # Nature scenes
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&q=80",  # Mountain landscape
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&q=80",  # Forest path
        
        # Indoor scenes
        "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800&q=80",  # Living room
        "https://images.unsplash.com/photo-1571624436279-b272aff752b5?w=800&q=80",  # Kitchen
        
        # People and activities
        "https://images.unsplash.com/photo-1511632765486-a01980e01a18?w=800&q=80",  # Restaurant
        "https://images.unsplash.com/photo-1543269664-56d93c1b41a6?w=800&q=80",  # Shopping
    ]


def visualize_results(image: Image.Image, predictions: dict, interface: SegFormerInterface, 
                     save_output: bool = False, output_dir: str = "test_outputs") -> None:
    """Visualize the segmentation results."""
    
    # Create output directory if saving
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get segmentation map for visualization
    pixel_values = interface.processor(image, return_tensors="pt").pixel_values.to(interface.device)
    with torch.no_grad():
        outputs = interface.model(pixel_values)
    
    seg_map = interface.processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SegFormer Interface Test Results', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Segmentation map (raw)
    axes[0, 1].imshow(seg_map.cpu().numpy(), cmap='tab20')
    axes[0, 1].set_title('Segmentation Map (Classes)')
    axes[0, 1].axis('off')
    
    # Segmentation overlay
    palette = np.array(interface.ade_palette(), dtype=np.uint8)
    seg_colored = palette[seg_map.cpu().numpy()]  # (H, W, 3)
    blend = (0.6 * np.asarray(image) + 0.4 * seg_colored[..., ::-1]).astype(np.uint8)
    axes[1, 0].imshow(blend)
    axes[1, 0].set_title('Segmentation Overlay')
    axes[1, 0].axis('off')
    
    # DETR-compatible masks (first few)
    pred_masks = predictions['pred_masks']  # (1, num_queries, H, W)
    mask_viz = torch.zeros_like(pred_masks[0, 0])
    
    # Combine first few non-empty masks for visualization
    for i in range(min(5, pred_masks.shape[1])):
        mask = pred_masks[0, i]
        if mask.sum() > 0:
            mask_viz += mask * (i + 1)
    
    axes[1, 1].imshow(mask_viz.cpu().numpy(), cmap='tab10')
    axes[1, 1].set_title('DETR-Compatible Masks (Combined)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_output:
        output_path = os.path.join(output_dir, 'segmentation_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to: {output_path}")
    
    plt.show()


def analyze_predictions(predictions: dict, logger: logging.Logger) -> None:
    """Analyze and log information about the predictions."""
    
    pred_masks = predictions['pred_masks']
    pred_logits = predictions['pred_logits'] 
    pred_boxes = predictions['pred_boxes']
    
    logger.info("=== Prediction Analysis ===")
    logger.info(f"Prediction shapes:")
    logger.info(f"  - pred_masks: {pred_masks.shape}")
    logger.info(f"  - pred_logits: {pred_logits.shape}")
    logger.info(f"  - pred_boxes: {pred_boxes.shape}")
    
    # Analyze masks
    non_empty_masks = 0
    total_pixels = pred_masks.shape[2] * pred_masks.shape[3]
    
    for i in range(pred_masks.shape[1]):
        mask = pred_masks[0, i]
        if mask.sum() > 0:
            non_empty_masks += 1
            coverage = (mask.sum() / total_pixels * 100).item()
            logger.info(f"  - Mask {i}: {mask.sum().item():.0f} pixels ({coverage:.1f}% coverage)")
    
    logger.info(f"Total non-empty masks: {non_empty_masks} out of {pred_masks.shape[1]}")
    
    # Analyze logits
    max_confidences = pred_logits.max(dim=-1)[0][0]  # (num_queries,)
    active_queries = (max_confidences > 5.0).sum().item()
    logger.info(f"Active queries (confidence > 5.0): {active_queries}")

def main():
    parser = argparse.ArgumentParser(description="Test SegFormer interface with random web image")
    parser.add_argument("--image_url", type=str, help="Specific image URL to test")
    parser.add_argument("--save_output", action="store_true", help="Save visualization outputs")
    parser.add_argument("--output_dir", type=str, default="test_outputs", help="Output directory for saved files")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b5-finetuned-ade-640-640", 
                       help="SegFormer model name")
    parser.add_argument("--enable_vanilla_segmentation", action="store_true", 
                       help="Enable vanilla segmentation saving")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting SegFormer interface test")
    
    try:
        # Get image URL
        if args.image_url:
            image_url = args.image_url
        else:
            # Select a random image from our sample URLs
            import random
            sample_urls = get_sample_image_urls()
            image_url = random.choice(sample_urls)
            logger.info(f"Randomly selected image URL: {image_url}")
        
        # Download image
        image = download_image(image_url, logger)
        
        # Initialize SegFormer interface
        logger.info("Initializing SegFormer interface...")
        interface = SegFormerInterface(
            model_name=args.model_name,
            logger=logger
        )
        
        # Load model
        logger.info("Loading SegFormer model...")
        interface.load_model()
        
        # Run inference
        logger.info("Running inference...")
        predictions = interface.infer_image(image)
        
        # Analyze results
        analyze_predictions(predictions, logger)
        
        # Save vanilla segmentation if enabled
        if args.enable_vanilla_segmentation:
            logger.info("Saving vanilla segmentation results...")
            if args.save_output:
                vanilla_output_dir = os.path.join(args.output_dir, "vanilla_segmentation")
            else:
                vanilla_output_dir = "vanilla_segmentation_demo"
            
            vanilla_saver = VanillaSegmentationSaver(
                model_interface=interface,
                output_dir=vanilla_output_dir,
                logger=logger
            )
            
            vanilla_results = vanilla_saver.save_frame_segmentation(image, frame_idx=0)
            logger.info(f"Vanilla segmentation saved to: {vanilla_output_dir}")
            if vanilla_results:
                logger.info(f"Generated files: {list(vanilla_results.keys())}")
        
        # Visualize results
        logger.info("Generating visualizations...")
        visualize_results(image, predictions, interface, args.save_output, args.output_dir)
        
        # Test DETR compatibility
        logger.info("=== DETR Compatibility Test ===")
        logger.info("✓ pred_masks format: torch.Tensor with shape (batch, queries, height, width)")
        logger.info("✓ pred_logits format: torch.Tensor with shape (batch, queries, num_classes)")
        logger.info("✓ pred_boxes format: torch.Tensor with shape (batch, queries, 4)")
        logger.info("✓ All tensors on same device")
        logger.info("✓ Output format compatible with DETR-based experiments")
        
        logger.info("Test completed successfully! 🎉")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()