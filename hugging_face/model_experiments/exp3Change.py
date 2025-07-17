#!/usr/bin/env python3
"""
exp3Change.py

Change Detection Experiment that processes raw image files and computes change detection success rates
based on blob segmentation across Concave, NoFill, and Convex categories.
Completely self-contained from raw images to final analysis.

FIXED VERSION:
- Fixed array indexing issues by ensuring all masks are boolean
- Enhanced error handling for mask operations
- Proper type conversion for regionprops and find_contours
- Robust mask handling throughout the pipeline
- Added proper analysis for Concave vs NoFill vs Convex categories
- Generates box plots showing "% Noticing Change" across categories

Usage:
    python exp3Change.py --model_interface segformer --images_dir /path/to/raw_images --output_dir /path/to/output [--resume]
"""

import argparse
import os
import sys
import json
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw
import glob
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from skimage.measure import label, regionprops, find_contours
from scipy.optimize import linear_sum_assignment

# Import model interfaces
from segformer.segformer_interface import SegFormerInterface, ModelInterface
from vanilla_segmentation import VanillaSegmentationSaver

torch.set_grad_enabled(False)

##############################################################################
# EXPERIMENT CLASS
##############################################################################

class ChangeDetectionExperiment:
    """
    Experiment 3: Change Detection Analysis
    
    This experiment:
    1. Takes a directory of raw image files as input
    2. Processes each image to detect and segment blobs
    3. Parses image names to extract Concave/NoFill/Convex categories
    4. Computes change detection success rates ("% Noticing Change") at various thresholds
    5. Generates box plots comparing categories across thresholds
    6. Saves results for each threshold comparison
    """
    
    def __init__(self, model_interface: ModelInterface, output_dir: str, logger: logging.Logger = None, 
                 enable_vanilla_segmentation: bool = True):
        self.model_interface = model_interface
        self.output_dir = output_dir
        self.enable_vanilla_segmentation = enable_vanilla_segmentation
        
        # Create output subdirectories FIRST (before logger is set up)
        self.results_dir = os.path.join(output_dir, "results")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.processed_images_dir = os.path.join(output_dir, "processed_images")
        self.threshold_results_dir = os.path.join(output_dir, "threshold_results")
        self.org_segmentation_dir = os.path.join(output_dir, "org_segmentation")
        
        for dir_path in [self.results_dir, self.plots_dir, self.logs_dir, 
                        self.processed_images_dir, self.threshold_results_dir, self.org_segmentation_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Now setup logger after logs_dir exists
        self.logger = logger or self._setup_logger()
        
        # Initialize vanilla segmentation saver if enabled
        self.vanilla_saver = None
        if self.enable_vanilla_segmentation:
            self.vanilla_saver = VanillaSegmentationSaver(
                model_interface=self.model_interface,
                output_dir=self.org_segmentation_dir,
                logger=self.logger
            )
            self.logger.info("Vanilla segmentation enabled for change detection experiment")
            
        self.logger.info(f"Initialized Change Detection Experiment with output dir: {output_dir}")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(self.logs_dir, f"change_detection_exp_{timestamp}.log")

        logger = logging.getLogger(f"change_detection_exp_{timestamp}")
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        logger.info(f"Logger initialized. Writing detailed log to {log_file_path}")
        return logger

    def run_full_experiment(self, images_dir: str, 
                          thresholds: List[int] = None, resume: bool = True) -> None:
        """Run the complete change detection experiment from raw images to final analysis."""
        self.logger.info("Starting full change detection experiment")
        
        if thresholds is None:
            thresholds = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]
        
        # Step 1: Find all image files in the input directory
        image_files = self._find_image_files(images_dir)
        if not image_files:
            self.logger.error(f"No image files found in {images_dir}")
            return
        
        self.logger.info(f"Found {len(image_files)} image files to process")
        
        # Step 2: Process each image to extract blob information and parse categories
        all_image_data = {}
        
        for image_file in image_files:
            image_name = Path(image_file).stem
            self.logger.info(f"Processing image: {image_name}")
            
            try:
                # Parse image category (Concave, NoFill, Convex)
                category_info = self._parse_image_category(image_name)
                
                if resume and self._is_image_already_processed(image_name):
                    self.logger.info(f"Image {image_name} already processed, loading existing data")
                    blob_data = self._load_existing_blob_data(image_name)
                else:
                    blob_data = self._process_single_image(image_file, image_name)
                
                if blob_data:
                    blob_data['category_info'] = category_info
                    all_image_data[image_name] = blob_data
                else:
                    self.logger.warning(f"No blob data extracted for image: {image_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process image {image_name}: {e}")
                continue
        
        # Step 3: Generate threshold comparisons and change detection analysis
        if all_image_data:
            self._analyze_change_detection_across_categories(all_image_data, thresholds)
        else:
            self.logger.warning("No image data available - skipping analysis")
        
        self.logger.info("Change detection experiment completed successfully")

    def _parse_image_category(self, image_name: str) -> Dict[str, Any]:
        """
        Parse image name to extract category information (Concave, NoFill, Convex).
        
        Expected naming patterns might include:
        - Images with "concave", "nofill", "convex" in the name
        - Or specific patterns that indicate the category
        """
        image_lower = image_name.lower()
        
        category = "Unknown"
        if "concave" in image_lower:
            category = "Concave"
        elif "nofill" in image_lower or "no_fill" in image_lower:
            category = "NoFill"
        elif "convex" in image_lower:
            category = "Convex"
        
        # Try to extract any numeric values that might be thresholds or ground truth
        import re
        numbers = re.findall(r'\d+', image_name)
        
        return {
            'category': category,
            'numbers_in_name': [int(n) for n in numbers],
            'parsed_successfully': category != "Unknown"
        }

    def _find_image_files(self, images_dir: str) -> List[str]:
        """Find all image files in the specified directory."""
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            pattern = os.path.join(images_dir, ext)
            image_files.extend(glob.glob(pattern))
        
        return sorted(image_files)

    def _is_image_already_processed(self, image_name: str) -> bool:
        """Check if an image has already been processed."""
        processed_dir = Path(self.processed_images_dir) / f"segformer_model_{image_name}"
        return processed_dir.exists() and (processed_dir / "mask").exists()

    def _load_existing_blob_data(self, image_name: str) -> Dict[str, Any]:
        """Load existing blob data from processed image directory."""
        try:
            processed_dir = Path(self.processed_images_dir) / f"segformer_model_{image_name}"
            mask_dir = processed_dir / "mask"
            
            if not mask_dir.exists():
                self.logger.warning(f"Mask directory does not exist: {mask_dir}")
                return None
            
            # Find mask files
            mask_files = list(mask_dir.glob("mask_*.png"))
            if not mask_files:
                self.logger.warning(f"No mask files found in {mask_dir}")
                return None
            
            # Load the first mask (assuming single image processing)
            mask_file = mask_files[0]
            mask_img = Image.open(mask_file).convert('L')
            mask_array = np.array(mask_img, dtype=np.uint8)
            binary_mask = (mask_array > 0).astype(np.float32)
            
            # Compute blob statistics
            blob_stats = self._compute_blob_statistics(binary_mask)
            change_detected = self._assess_change_detection_success(binary_mask, image_name)
            
            return {
                'mask': binary_mask,
                'blob_stats': blob_stats,
                'change_detected': change_detected,
                'processed_dir': str(processed_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load existing blob data for {image_name}: {e}")
            return None

    def _assess_change_detection_success(self, mask: np.ndarray, image_name: str) -> bool:
        """
        Assess whether change detection was successful for this image.
        
        This is a simplified assessment - in practice, this would compare against
        ground truth or use more sophisticated metrics.
        """
        if mask is None or mask.sum() == 0:
            return False
        
        # Simple heuristic: if we detected a reasonable amount of change (blob area)
        total_pixels = mask.shape[0] * mask.shape[1]
        change_ratio = mask.sum() / total_pixels
        
        # Consider change detected if between 1% and 50% of image
        return 0.01 <= change_ratio <= 0.5

    def _process_single_image(self, image_path: str, image_name: str) -> Dict[str, Any]:
        """Process a single image to detect and segment blobs."""
        
        # Load model if not already loaded
        if not hasattr(self.model_interface, 'model') or self.model_interface.model is None:
            self.logger.info("Loading model...")
            self.model_interface.load_model()
        
        # Setup output directories for this image
        output_base = os.path.join(self.processed_images_dir, f"segformer_model_{image_name}")
        dirs = {
            "blobs": os.path.join(output_base, "blobs"),
            "collage": os.path.join(output_base, "collage"),
            "mask": os.path.join(output_base, "mask"),
            "proc": os.path.join(output_base, "processed"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        # Load image and ensure RGB format
        try:
            frame = np.array(Image.open(image_path).convert('RGB'))
            H, W, _ = frame.shape
            self.logger.info(f"Image {image_name} shape: {H}x{W} RGB")
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            return None

        # Detect blob using intensity thresholding
        blob = self._detect_blob(frame)
        if blob is None:
            self.logger.warning(f"No blob found in image {image_name}")
            return None

        # Save blob overlay
        self._save_blob_overlay(frame, blob, dirs['blobs'], image_name)

        # Run model inference to get segmentation masks
        try:
            candidates = self._run_model_inference(frame, H, W)
            self.logger.info(f"Generated {len(candidates)} candidate masks from model.")
        except Exception as e:
            self.logger.error(f"Model inference failed for {image_name}: {e}")
            return None

        # Choose best mask that matches the detected blob
        chosen_mask = self._choose_best_mask(blob, candidates)
        
        if chosen_mask is not None and chosen_mask.sum() > 0:
            mask_file = os.path.join(dirs['mask'], f"mask_{image_name}.png")
            # Ensure mask is boolean and convert to uint8 for saving
            mask_bool = chosen_mask.astype(bool)
            mask_uint8 = (mask_bool.astype(np.uint8) * 255)
            Image.fromarray(mask_uint8).save(mask_file)
            self.logger.info(f"Saved chosen mask to {mask_file}")
        else:
            self.logger.warning(f"No suitable model mask found for {image_name}")
            chosen_mask = blob.astype(bool) if blob is not None else None  # Fall back to detected blob

        # Generate collage of top candidate masks
        if candidates:
            self._save_mask_collage(frame, blob, candidates, dirs['collage'], image_name)

        # Generate final overlay with polygon
        self._save_final_overlay(frame, chosen_mask, dirs['proc'], image_name, W, H)

        # Compute blob statistics and change detection success
        blob_stats = self._compute_blob_statistics(chosen_mask)
        change_detected = self._assess_change_detection_success(chosen_mask, image_name)
        
        # Save vanilla segmentation if enabled
        if self.enable_vanilla_segmentation and self.vanilla_saver is not None:
            try:
                self.vanilla_saver.save_frame_segmentation(frame, frame_idx=0)  # Use 0 for single images
            except Exception as e:
                self.logger.warning(f"Failed to save vanilla segmentation for image {image_name}: {e}")

        return {
            'mask': chosen_mask,
            'blob_stats': blob_stats,
            'change_detected': change_detected,
            'processed_dir': output_base
        }

    def _detect_blob(self, frame: np.ndarray, thresholds: Tuple[int, ...] = (30, 15, 5)) -> np.ndarray:
        """Detect blob using intensity thresholding."""
        gray = frame.sum(axis=2)
        for thr in thresholds:
            labeled = label(gray > thr, connectivity=2)
            regs = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)
            if regs:
                self.logger.debug(f"Blob detected with threshold {thr} (area={regs[0].area})")
                # Ensure we return a boolean array
                blob_mask = (labeled == regs[0].label).astype(bool)
                return blob_mask
            else:
                self.logger.debug(f"No blob found at threshold {thr}")
        return None

    def _save_blob_overlay(self, frame: np.ndarray, blob: np.ndarray, output_dir: str, image_name: str) -> None:
        """Save blob overlay image."""
        overlay = frame.copy()
        # Ensure blob is boolean for indexing
        blob_bool = blob.astype(bool) if blob is not None else np.zeros_like(frame[:,:,0], dtype=bool)
        overlay[blob_bool] = [255, 0, 0]
        blob_file = os.path.join(output_dir, f"{image_name}_blobs.png")
        Image.fromarray(overlay).save(blob_file)

    def _run_model_inference(self, frame: np.ndarray, H: int, W: int) -> List[np.ndarray]:
        """Run model inference to get segmentation candidates."""
        # Convert frame to PIL Image for model inference
        pil_image = Image.fromarray(frame).convert('RGB')
        
        # Run inference using the model interface
        predictions = self.model_interface.infer_image(pil_image)
        
        # Extract masks from predictions
        pred_masks = predictions['pred_masks']  # Shape: (1, num_queries, H, W)
        
        candidates = []
        for i in range(pred_masks.shape[1]):
            # Get mask for this query
            mask = pred_masks[0, i].cpu().numpy()  # (H, W)
            
            # Threshold to get binary mask
            binary_mask = (mask > 0.5).astype(bool)  # Use bool instead of float32
            
            # Connected component analysis to get individual blobs
            labeled = label(binary_mask, connectivity=2)
            for lbl in range(1, labeled.max() + 1):
                component_mask = (labeled == lbl).astype(bool)  # Use bool instead of float32
                if component_mask.sum() > 0:  # Only add non-empty masks
                    candidates.append(component_mask)
        
        return candidates

    def _choose_best_mask(self, blob: np.ndarray, candidates: List[np.ndarray]) -> np.ndarray:
        """Choose the best mask from candidates based on IoU with detected blob."""
        if not candidates:
            return None
            
        best_iou = -1
        best_mask = None
        
        for candidate in candidates:
            iou = self._compute_iou(blob, candidate)
            if iou > best_iou:
                best_iou = iou
                best_mask = candidate
        
        # Ensure the returned mask is boolean
        if best_mask is not None and best_iou > 0.1:
            return best_mask.astype(bool)
        else:
            return None  # Minimum IoU threshold

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        try:
            # Handle None masks
            if mask1 is None or mask2 is None:
                return 0.0
            
            # Ensure masks are boolean
            mask1_binary = mask1.astype(bool) if mask1 is not None else np.zeros_like(mask2, dtype=bool)
            mask2_binary = mask2.astype(bool) if mask2 is not None else np.zeros_like(mask1, dtype=bool)
            
            intersection = np.logical_and(mask1_binary, mask2_binary).sum()
            union = np.logical_or(mask1_binary, mask2_binary).sum()
            
            if union == 0:
                return 0.0
            
            return float(intersection / union)
            
        except Exception as e:
            self.logger.warning(f"Error computing IoU: {e}")
            return 0.0

    def _save_mask_collage(self, frame: np.ndarray, blob: np.ndarray, candidates: List[np.ndarray], 
                          output_dir: str, image_name: str) -> None:
        """Save collage of top candidate masks."""
        if not candidates:
            return
            
        ious = [self._compute_iou(blob, mask) for mask in candidates]
        best_indices = np.argsort(ious)[::-1][:10]  # Top 10
        
        fig, axes = plt.subplots(1, min(10, len(best_indices)), figsize=(25, 3), dpi=100)
        if len(best_indices) == 1:
            axes = [axes]
            
        for i, idx in enumerate(best_indices):
            if i >= len(axes):
                break
                
            overlay = frame.copy()
            # Ensure masks are boolean for indexing
            blob_bool = blob.astype(bool) if blob is not None else np.zeros_like(frame[:,:,0], dtype=bool)
            candidate_bool = candidates[idx].astype(bool) if candidates[idx] is not None else np.zeros_like(frame[:,:,0], dtype=bool)
            
            overlay[blob_bool] = [0, 255, 0]  # Green for ground truth
            overlay[candidate_bool] = [255, 0, 0]  # Red for candidate
            
            axes[i].imshow(overlay)
            axes[i].set_title(f"#{idx}\nIoU: {ious[idx]:.3f}", fontsize=8)
            axes[i].axis('off')
        
        fig.suptitle(f"{image_name} - Top 10 Mask Candidates", fontsize=14)
        collage_file = os.path.join(output_dir, f"{image_name}_collage.png")
        fig.savefig(collage_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _save_final_overlay(self, frame: np.ndarray, mask: np.ndarray, output_dir: str, 
                           image_name: str, W: int, H: int) -> None:
        """Save final overlay with polygon."""
        final = Image.fromarray(frame.copy())
        if mask is not None and mask.sum() > 0:
            # Ensure mask is boolean and convert to uint8 for find_contours
            mask_bool = mask.astype(bool)
            mask_uint8 = mask_bool.astype(np.uint8)
            
            # Find contours and create polygon
            contours = find_contours(mask_uint8, 0.5)
            if contours:
                biggest_contour = max(contours, key=len)
                polygon_points = [(p[1], p[0]) for p in biggest_contour]
                
                draw = ImageDraw.Draw(final, 'RGBA')
                draw.polygon(polygon_points, fill=(255, 0, 0, 120))
                
                # Add centroid text
                centroid_x = np.mean([p[0] for p in polygon_points])
                centroid_y = np.mean([p[1] for p in polygon_points])
                draw.text((centroid_x, centroid_y), 'Blob', fill=(255, 255, 255, 255))
        
        final_file = os.path.join(output_dir, f"{image_name}_overlay.png")
        final.save(final_file)

    def _compute_blob_statistics(self, mask: np.ndarray) -> Dict[str, float]:
        """Compute statistics for a blob mask."""
        if mask is None or mask.sum() == 0:
            return {'area': 0.0, 'centroid_x': 0.0, 'centroid_y': 0.0, 'perimeter': 0.0}
        
        try:
            # Ensure mask is boolean and convert to uint8 for regionprops
            mask_bool = mask.astype(bool)
            mask_uint8 = mask_bool.astype(np.uint8)
            
            labeled = label(mask_uint8, connectivity=2)
            props = regionprops(labeled)
            
            if not props:
                return {'area': 0.0, 'centroid_x': 0.0, 'centroid_y': 0.0, 'perimeter': 0.0}
            
            prop = props[0]  # Largest component
            centroid_y, centroid_x = prop.centroid
            
            return {
                'area': float(prop.area),
                'centroid_x': float(centroid_x),
                'centroid_y': float(centroid_y),
                'perimeter': float(prop.perimeter)
            }
        
        except Exception as e:
            self.logger.warning(f"Error computing blob statistics: {e}")
            return {'area': 0.0, 'centroid_x': 0.0, 'centroid_y': 0.0, 'perimeter': 0.0}

    def _analyze_change_detection_across_categories(self, all_image_data: Dict[str, Dict[str, Any]], 
                                                  thresholds: List[int]) -> None:
        """Analyze change detection success rates across Concave, NoFill, and Convex categories."""
        self.logger.info("Analyzing change detection across categories (Concave, NoFill, Convex)")
        
        # Group images by category
        category_data = {'Concave': [], 'NoFill': [], 'Convex': []}
        
        for image_name, image_data in all_image_data.items():
            category_info = image_data.get('category_info', {})
            category = category_info.get('category', 'Unknown')
            
            if category in category_data:
                category_data[category].append({
                    'image_name': image_name,
                    'change_detected': image_data.get('change_detected', False),
                    'blob_stats': image_data.get('blob_stats', {}),
                    'category_info': category_info
                })
            else:
                self.logger.warning(f"Unknown category '{category}' for image {image_name}")
        
        # Log category counts
        for category, data_list in category_data.items():
            self.logger.info(f"Category '{category}': {len(data_list)} images")
        
        # For each threshold, compute change detection success rates
        threshold_results = {}
        
        for threshold in thresholds:
            self.logger.info(f"Analyzing threshold: {threshold}")
            
            threshold_results[threshold] = {}
            
            for category, data_list in category_data.items():
                if not data_list:
                    threshold_results[threshold][category] = []
                    continue
                
                # Compute success rates for this category at this threshold
                success_rates = []
                
                for image_data in data_list:
                    # Apply threshold-based logic to determine success
                    blob_stats = image_data['blob_stats']
                    area = blob_stats.get('area', 0)
                    
                    # Success criteria: detected change AND area meets threshold requirements
                    change_detected = image_data['change_detected']
                    area_meets_threshold = area >= (threshold * 10)  # Scale threshold
                    
                    success = change_detected and area_meets_threshold
                    success_rates.append(1.0 if success else 0.0)
                
                threshold_results[threshold][category] = success_rates
        
        # Generate box plots and analysis
        self._generate_category_box_plots(threshold_results, thresholds)
        
        # Save detailed results
        self._save_category_analysis_results(threshold_results, category_data)

    def _generate_category_box_plots(self, threshold_results: Dict[int, Dict[str, List[float]]], 
                                   thresholds: List[int]) -> None:
        """Generate box plots showing % Noticing Change across categories for different thresholds."""
        
        # Create figure with subplots for each threshold
        n_thresholds = len(thresholds)
        cols = min(4, n_thresholds)
        rows = (n_thresholds + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_thresholds == 1:
            axes = [axes]
        elif rows == 1:
            pass  # axes is already 1D
        else:
            axes = axes.flatten()
        
        categories = ['Concave', 'NoFill', 'Convex']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
        
        for i, threshold in enumerate(thresholds):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Prepare data for box plot
            box_data = []
            labels = []
            
            for category in categories:
                success_rates = threshold_results[threshold].get(category, [])
                if success_rates:
                    # Convert to percentages
                    percentages = [rate * 100 for rate in success_rates]
                    box_data.append(percentages)
                    labels.append(f"{category}\n(n={len(success_rates)})")
                else:
                    box_data.append([0])  # Empty data
                    labels.append(f"{category}\n(n=0)")
            
            # Create box plot
            if any(len(data) > 0 for data in box_data):
                bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            ax.set_title(f'Threshold {threshold}', fontsize=12, fontweight='bold')
            ax.set_ylabel('% Noticing Change', fontsize=10)
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)
            
            # Add mean values as text
            for j, (category, data) in enumerate(zip(categories, box_data)):
                if data and len(data) > 0:
                    mean_val = np.mean(data)
                    ax.text(j+1, mean_val + 2, f'{mean_val:.1f}%', 
                           ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # Hide unused subplots
        for i in range(n_thresholds, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Change Detection Success Rates by Category and Threshold', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        # Save plot
        box_plot_path = os.path.join(self.plots_dir, "category_box_plots.png")
        plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Saved category box plots to {box_plot_path}")
        
        # Generate summary plot across all thresholds
        self._generate_summary_category_plot(threshold_results, thresholds)

    def _generate_summary_category_plot(self, threshold_results: Dict[int, Dict[str, List[float]]], 
                                      thresholds: List[int]) -> None:
        """Generate summary plot showing mean % Noticing Change across all thresholds."""
        
        categories = ['Concave', 'NoFill', 'Convex']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Compute mean success rates for each category across thresholds
        category_means = {category: [] for category in categories}
        category_stds = {category: [] for category in categories}
        
        for threshold in thresholds:
            for category in categories:
                success_rates = threshold_results[threshold].get(category, [])
                if success_rates:
                    percentages = [rate * 100 for rate in success_rates]
                    category_means[category].append(np.mean(percentages))
                    category_stds[category].append(np.std(percentages))
                else:
                    category_means[category].append(0)
                    category_stds[category].append(0)
        
        # Create line plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, category in enumerate(categories):
            means = category_means[category]
            stds = category_stds[category]
            
            ax.plot(thresholds, means, 'o-', color=colors[i], linewidth=2, 
                   markersize=8, label=category)
            ax.errorbar(thresholds, means, yerr=stds, color=colors[i], 
                       capsize=5, alpha=0.7)
        
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('% Noticing Change (Mean)', fontsize=12)
        ax.set_title('Change Detection Success Rates by Category Across Thresholds', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        # Save plot
        summary_plot_path = os.path.join(self.plots_dir, "category_summary_plot.png")
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Saved category summary plot to {summary_plot_path}")

    def _save_category_analysis_results(self, threshold_results: Dict[int, Dict[str, List[float]]], 
                                      category_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Save detailed category analysis results."""
        
        # Prepare summary data
        summary_data = []
        
        for threshold in sorted(threshold_results.keys()):
            for category in ['Concave', 'NoFill', 'Convex']:
                success_rates = threshold_results[threshold].get(category, [])
                if success_rates:
                    percentages = [rate * 100 for rate in success_rates]
                    summary_data.append({
                        'threshold': threshold,
                        'category': category,
                        'mean_success_rate': np.mean(percentages),
                        'std_success_rate': np.std(percentages),
                        'n_images': len(success_rates),
                        'success_rates': percentages
                    })
                else:
                    summary_data.append({
                        'threshold': threshold,
                        'category': category,
                        'mean_success_rate': 0.0,
                        'std_success_rate': 0.0,
                        'n_images': 0,
                        'success_rates': []
                    })
        
        # Save to CSV
        csv_data = []
        for entry in summary_data:
            csv_data.append({
                'threshold': entry['threshold'],
                'category': entry['category'],
                'mean_success_rate': entry['mean_success_rate'],
                'std_success_rate': entry['std_success_rate'],
                'n_images': entry['n_images']
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.results_dir, "category_analysis_summary.csv")
        df.to_csv(csv_path, index=False)
        
        # Save detailed JSON
        json_path = os.path.join(self.results_dir, "category_analysis_detailed.json")
        
        detailed_results = {
            'threshold_results': threshold_results,
            'category_counts': {cat: len(data) for cat, data in category_data.items()},
            'summary_statistics': summary_data
        }
        
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        self.logger.info(f"Saved category analysis to {csv_path} and {json_path}")

##############################################################################
# MAIN FUNCTION
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Change Detection Experiment - Process raw images and analyze blob segmentation across categories")
    parser.add_argument("--model_interface", type=str, default="segformer",
                      choices=["segformer"], help="Model interface to use")
    parser.add_argument("--images_dir", type=str, required=False,
                      default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/hugging_face/model_experiments/exp3Change_files",
                      help="Directory containing raw image files")
    parser.add_argument("--output_dir", type=str, required=False,
                      default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/hugging_face/model_experiments/segformer/exp3Change",
                      help="Output directory for results and processed data")
    parser.add_argument("--thresholds", type=int, nargs='+', 
                      default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20],
                      help="List of thresholds to analyze (default: 1 2 3 4 5 6 8 10 12 14 16 18 20)")
    parser.add_argument("--resume", action="store_true", default=True,
                      help="Resume processing from checkpoints (default: True)")
    parser.add_argument("--no_resume", action="store_true", default=False,
                      help="Start processing from scratch, ignoring checkpoints")
    
    args = parser.parse_args()
    
    # Handle resume logic
    resume = args.resume and not args.no_resume
    
    # Validate inputs
    if not os.path.isdir(args.images_dir):
        print(f"Error: Images directory '{args.images_dir}' does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model interface
    if args.model_interface == "segformer":
        model_interface = SegFormerInterface()
    else:
        raise ValueError(f"Unknown model interface: {args.model_interface}")
    
    # Run experiment
    experiment = ChangeDetectionExperiment(model_interface, args.output_dir)
    
    try:
        experiment.run_full_experiment(
            images_dir=args.images_dir,
            thresholds=args.thresholds,
            resume=resume
        )
        print("Change detection experiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()