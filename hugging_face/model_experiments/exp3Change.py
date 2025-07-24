#!/usr/bin/env python3
"""
exp3Change.py

Change Detection Experiment that processes raw image pairs and computes change detection success rates
based on area changes between before/after blob segmentations across Concave, NoFill, and Convex categories.
Follows the exact analysis pipeline from main_extract_mistake_score_and_plot.py but uses general model interface.

FEATURES:
- Processes image pairs (_init and _out) like the original analysis
- Uses general model interface (SegFormer) instead of DETR
- Computes area change ratios between before/after masks
- Generates threshold-based detection analysis
- Creates bar plots with diagonal hatches and SEM error bars
- Categorizes images as concave, concave_nofill, convex, no_change
- Saves original segmentation overlays for each image
- Verbose logging and organized output structure

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
from typing import List, Dict, Any, Tuple, Optional
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

torch.set_grad_enabled(False)

# Plotting parameters (matching main_extract_mistake_score_and_plot.py)
LABEL_FONTSIZE = 21
TICKS_FONTSIZE = 19
HIGH_DPI = 200
BAR_WIDTH = 1.0
LEFT_MARGIN = 0.5

##############################################################################
# EXPERIMENT CLASS
##############################################################################

class ChangeDetectionExperiment:
    """
    Experiment 3: Change Detection Analysis (Following Original Pipeline)
    
    This experiment:
    1. Finds image pairs (_init and _out) in the input directory
    2. Processes each image to detect and segment blobs using general model interface
    3. Saves original segmentation overlays (without blob selection) for each image
    4. Computes area change ratios between before/after masks
    5. Categorizes images by type (concave, concave_nofill, convex, no_change)
    6. Analyzes detection rates across multiple thresholds
    7. Generates bar plots with exact styling from original analysis
    8. Saves detailed results and visualizations
    """
    
    def __init__(self, model_interface: ModelInterface, output_dir: str, logger: logging.Logger = None):
        self.model_interface = model_interface
        self.output_dir = output_dir
        
        # Create only the directories we actually use
        self.logs_dir = os.path.join(output_dir, "logs")
        self.processed_images_dir = os.path.join(output_dir, "processed_images")
        self.threshold_results_dir = os.path.join(output_dir, "threshold_results")
        
        # Create only necessary directories
        for dir_path in [self.logs_dir, self.processed_images_dir, self.threshold_results_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Setup logger
        self.logger = logger or self._setup_logger()
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

    def run_full_experiment(self, images_dir: str, resume: bool = True) -> None:
        """Run the complete change detection experiment following the original analysis pipeline."""
        self.logger.info("Starting full change detection experiment (following original pipeline)")
        
        # Step 1: Find image pairs (_init and _out) like in original analysis
        image_pairs = self._find_image_pairs(images_dir)
        if not image_pairs:
            self.logger.error(f"No valid image pairs found in {images_dir}")
            return
        
        self.logger.info(f"Found {len(image_pairs)} image pairs to process")
        
        # Step 2: Process each image pair to extract blob information
        pair_details = []
        
        for base_name, init_path, out_path in image_pairs:
            self.logger.info(f"Processing image pair: {base_name}")
            
            try:
                # Process both init and out images
                init_data = self._process_single_image(init_path, f"{base_name}_init", resume)
                out_data = self._process_single_image(out_path, f"{base_name}_out", resume)
                
                if init_data and out_data:
                    # Compute area change ratio
                    area_change_data = self._compute_area_change(init_data, out_data, base_name)
                    if area_change_data:
                        pair_details.append(area_change_data)
                else:
                    self.logger.warning(f"Failed to process one or both images for pair: {base_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process image pair {base_name}: {e}")
                continue
        
        # Step 3: Perform threshold analysis (following original pipeline exactly)
        if pair_details:
            self._perform_threshold_analysis(pair_details)
        else:
            self.logger.warning("No valid image pair data - skipping threshold analysis")
        
        self.logger.info("Change detection experiment completed successfully")

    def _find_image_pairs(self, images_dir: str) -> List[Tuple[str, str, str]]:
        """
        Find image pairs following the original analysis pattern.
        Looks for _init and _out pairs, excluding catch_shape images.
        """
        pairs = []
        
        # Get all files in directory
        all_files = os.listdir(images_dir)
        
        # Find all _init files
        init_files = [f for f in all_files if f.endswith('_init.png') or f.endswith('_init.jpg') or f.endswith('_init.jpeg')]
        
        for init_file in init_files:
            if 'catch_shape' in init_file:
                continue
                
            # Extract base name
            base_with_ext = init_file[:-9]  # Remove '_init.png' or similar
            base_name = os.path.splitext(base_with_ext)[0]
            
            # Look for corresponding _out file
            init_path = os.path.join(images_dir, init_file)
            
            # Try different extensions for out file
            out_file = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_out = f"{base_name}_out{ext}"
                if potential_out in all_files:
                    out_file = potential_out
                    break
            
            if out_file:
                out_path = os.path.join(images_dir, out_file)
                pairs.append((base_name, init_path, out_path))
                self.logger.debug(f"Found pair: {base_name} -> {init_file} & {out_file}")
            else:
                self.logger.warning(f"No matching _out file found for {init_file}")
        
        return sorted(pairs)

    def _process_single_image(self, image_path: str, image_name: str, resume: bool) -> Optional[Dict[str, Any]]:
        """
        Process a single image following the main_segment_blobs.py pipeline.
        Returns mask data and blob statistics.
        """
        self.logger.info(f"  Processing image: {image_name}")
        
        # Check if already processed
        if resume and self._is_image_already_processed(image_name):
            self.logger.info(f"  Image {image_name} already processed, loading existing data")
            return self._load_existing_image_data(image_name)
        
        # Load model if not already loaded
        if not hasattr(self.model_interface, 'model') or self.model_interface.model is None:
            self.logger.info("Loading model...")
            self.model_interface.load_model()
        
        # Setup output directories for this image (following main_segment_blobs.py structure)
        model_prefix = "segformer_model"
        output_base = os.path.join(self.processed_images_dir, f"{model_prefix}_{image_name}")
        dirs = {
            "blobs": os.path.join(output_base, "frames_blobs"),
            "collage": os.path.join(output_base, "frames_collage"), 
            "mask": os.path.join(output_base, "frames_masks_nonmem"),
            "proc": os.path.join(output_base, "frames_processed"),
            "original_seg": os.path.join(output_base, "original_segmentation"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        # Load image and ensure RGB format
        try:
            frame = np.array(Image.open(image_path).convert('RGB'))
            H, W, _ = frame.shape
            self.logger.info(f"  Image {image_name} shape: {H}x{W} RGB")
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            return None

        # Save original segmentation overlay (like demo_interface.py)
        self._save_original_segmentation_overlay(frame, dirs['original_seg'], image_name)

        # Detect blob using intensity thresholding (following main_segment_blobs.py)
        blob = self._detect_blob(frame)
        if blob is None:
            self.logger.warning(f"  No blob found in image {image_name}")
            return None

        # Save blob overlay
        self._save_blob_overlay(frame, blob, dirs['blobs'], image_name)

        # Run model inference to get segmentation candidates
        try:
            candidates = self._run_model_inference(frame, H, W)
            self.logger.info(f"  Generated {len(candidates)} candidate masks from model.")
        except Exception as e:
            self.logger.error(f"Model inference failed for {image_name}: {e}")
            return None

        # Choose best mask that matches the detected blob
        chosen_mask, best_cost = self._choose_best_mask(blob, candidates)
        
        if chosen_mask is not None and chosen_mask.sum() > 0:
            mask_file = os.path.join(dirs['mask'], f"mask_{image_name}.png")
            mask_uint8 = (chosen_mask.astype(np.uint8) * 255)
            Image.fromarray(mask_uint8).save(mask_file)
            self.logger.info(f"  Chosen mask cost={best_cost:.3f} -> {mask_file}")
        else:
            self.logger.warning(f"  No suitable model mask found for {image_name}, using detected blob")
            chosen_mask = blob.astype(bool) if blob is not None else None

        # Generate collage of top candidate masks (following main_segment_blobs.py)
        if candidates:
            self._save_mask_collage(frame, blob, candidates, dirs['collage'], image_name)

        # Generate final overlay with polygon
        self._save_final_overlay(frame, chosen_mask, dirs['proc'], image_name, W, H)

        # Compute blob statistics
        blob_stats = self._compute_blob_statistics(chosen_mask)

        return {
            'mask': chosen_mask,
            'blob_stats': blob_stats,
            'mask_file': mask_file if chosen_mask is not None and chosen_mask.sum() > 0 else None,
            'processed_dir': output_base
        }

    def _save_original_segmentation_overlay(self, frame: np.ndarray, output_dir: str, image_name: str) -> None:
        """
        Save original segmentation overlay (without blob selection) like demo_interface.py.
        This shows what the raw model segmentation looks like.
        """
        try:
            # Convert frame to PIL Image for model inference
            pil_image = Image.fromarray(frame).convert('RGB')
            
            # Get raw segmentation map from model
            pixel_values = self.model_interface.processor(pil_image, return_tensors="pt").pixel_values.to(self.model_interface.device)
            with torch.no_grad():
                outputs = self.model_interface.model(pixel_values)
            
            seg_map = self.model_interface.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[pil_image.size[::-1]]
            )[0]  # Shape: (H, W)
            
            # Create colored segmentation overlay (following demo_interface.py)
            palette = np.array(self.model_interface.ade_palette(), dtype=np.uint8)
            seg_colored = palette[seg_map.cpu().numpy()]  # (H, W, 3)
            
            # Create blend overlay (60% original, 40% segmentation)
            blend = (0.6 * np.asarray(pil_image) + 0.4 * seg_colored[..., ::-1]).astype(np.uint8)
            
            # Save the original segmentation overlay
            overlay_file = os.path.join(output_dir, f"{image_name}_original_segmentation.png")
            Image.fromarray(blend).save(overlay_file)
            
            # Also save the raw segmentation map
            seg_map_file = os.path.join(output_dir, f"{image_name}_segmentation_map.png")
            Image.fromarray(seg_colored).save(seg_map_file)
            
            self.logger.info(f"    Saved original segmentation overlay -> {overlay_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save original segmentation overlay for {image_name}: {e}")

    def _detect_blob(self, frame: np.ndarray, thresholds: Tuple[int, ...] = (30, 15, 5)) -> Optional[np.ndarray]:
        """Detect blob using intensity thresholding (following main_segment_blobs.py)."""
        gray = frame.sum(axis=2)
        for thr in thresholds:
            labeled = label(gray > thr, connectivity=2)
            regs = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)
            if regs:
                self.logger.debug(f"    Blob detected with threshold {thr} (area={regs[0].area})")
                blob_mask = (labeled == regs[0].label).astype(bool)
                return blob_mask
            else:
                self.logger.debug(f"    No blob found at threshold {thr}")
        return None

    def _save_blob_overlay(self, frame: np.ndarray, blob: np.ndarray, output_dir: str, image_name: str) -> None:
        """Save blob overlay image (following main_segment_blobs.py)."""
        overlay = frame.copy()
        blob_bool = blob.astype(bool) if blob is not None else np.zeros_like(frame[:,:,0], dtype=bool)
        overlay[blob_bool] = [255, 0, 0]
        blob_file = os.path.join(output_dir, f"{image_name}_blobs.png")
        Image.fromarray(overlay).save(blob_file)
        self.logger.info(f"    Saved blob overlay -> {blob_file}")

    def _run_model_inference(self, frame: np.ndarray, H: int, W: int) -> List[np.ndarray]:
        """Run model inference and split connected components (following general interface pattern)."""
        pil_image = Image.fromarray(frame).convert('RGB')
        
        # Run inference using the model interface
        predictions = self.model_interface.infer_image(pil_image)
        pred_masks = predictions['pred_masks']  # Shape: (1, num_queries, H, W)
        
        candidates = []
        for i in range(pred_masks.shape[1]):
            # Get mask for this query
            mask = pred_masks[0, i].cpu().numpy()  # (H, W)
            
            # Threshold to get binary mask
            binary_mask = (mask > 0.5).astype(bool)
            
            # Split into connected components
            labeled = label(binary_mask, connectivity=2)
            for lbl in range(1, labeled.max() + 1):
                component_mask = (labeled == lbl).astype(bool)
                if component_mask.sum() > 0:
                    candidates.append(component_mask)
        
        return candidates

    def _choose_best_mask(self, blob: np.ndarray, candidates: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Choose the best mask from candidates based on IoU with detected blob (following main_segment_blobs.py)."""
        if not candidates:
            return None, None
        
        costs = np.array([-self._compute_iou(blob, mask) for mask in candidates], dtype=np.float32)
        best_idx = int(costs.argmin())
        best_cost = float(costs[best_idx])
        
        return candidates[best_idx].astype(bool), best_cost

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        try:
            if mask1 is None or mask2 is None:
                return 0.0
            
            mask1_binary = mask1.astype(bool)
            mask2_binary = mask2.astype(bool)
            
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
        """Save collage of top candidate masks (following main_segment_blobs.py)."""
        if not candidates:
            return
            
        ious = [-self._compute_iou(blob, mask) for mask in candidates]
        best_indices = np.argsort(ious)[:10]  # Top 10
        
        fig, axes = plt.subplots(1, 10, figsize=(25, 3), dpi=100)
        
        for i, idx in enumerate(best_indices):
            overlay = frame.copy()
            blob_bool = blob.astype(bool) if blob is not None else np.zeros_like(frame[:,:,0], dtype=bool)
            candidate_bool = candidates[idx].astype(bool)
            
            overlay[blob_bool] = [0, 255, 0]  # Green for ground truth
            overlay[candidate_bool] = [255, 0, 0]  # Red for candidate
            
            axes[i].imshow(overlay)
            axes[i].set_title(f"#{idx}\n{ious[idx]:.3f}", fontsize=6)
            axes[i].axis('off')
        
        collage_file = os.path.join(output_dir, f"{image_name}_collage.png")
        fig.savefig(collage_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        self.logger.info(f"    Saved collage -> {collage_file}")

    def _save_final_overlay(self, frame: np.ndarray, mask: np.ndarray, output_dir: str, 
                           image_name: str, W: int, H: int) -> None:
        """Save final overlay with polygon (following main_segment_blobs.py)."""
        final = Image.fromarray(frame.copy())
        if mask is not None and mask.sum() > 0:
            # Create polygon from mask contours
            contours = find_contours(mask.astype(np.uint8), 0.5)
            if contours:
                biggest_contour = max(contours, key=len)
                # Convert to polygon points relative to center
                cx, cy = W / 2.0, H / 2.0
                polygon_points = [(p[1] - cx, p[0] - cy) for p in biggest_contour]
                
                # Convert back to absolute coordinates for drawing
                abs_points = [(x + cx, y + cy) for x, y in polygon_points]
                
                draw = ImageDraw.Draw(final, 'RGBA')
                draw.polygon(abs_points, fill=(255, 0, 0, 120))
                
                # Add centroid text
                if abs_points:
                    centroid_x = sum(p[0] for p in abs_points) / len(abs_points)
                    centroid_y = sum(p[1] for p in abs_points) / len(abs_points)
                    draw.text((centroid_x, centroid_y), 'Blob', fill=(255, 255, 255, 255))
        
        final_file = os.path.join(output_dir, f"{image_name}_overlay.png")
        final.save(final_file)
        self.logger.info(f"    Saved final overlay -> {final_file}")

    def _compute_blob_statistics(self, mask: np.ndarray) -> Dict[str, float]:
        """Compute statistics for a blob mask."""
        if mask is None or mask.sum() == 0:
            return {'area': 0.0, 'centroid_x': 0.0, 'centroid_y': 0.0, 'perimeter': 0.0}
        
        try:
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

    def _is_image_already_processed(self, image_name: str) -> bool:
        """Check if an image has already been processed."""
        processed_dir = Path(self.processed_images_dir) / f"segformer_model_{image_name}"
        return processed_dir.exists() and (processed_dir / "frames_masks_nonmem").exists()

    def _load_existing_image_data(self, image_name: str) -> Optional[Dict[str, Any]]:
        """Load existing image data from processed directory."""
        try:
            processed_dir = Path(self.processed_images_dir) / f"segformer_model_{image_name}"
            mask_dir = processed_dir / "frames_masks_nonmem"
            
            if not mask_dir.exists():
                return None
            
            # Find mask file
            mask_files = list(mask_dir.glob("mask_*.png"))
            if not mask_files:
                return None
            
            # Load the mask
            mask_file = mask_files[0]
            mask_img = Image.open(mask_file).convert('L')
            mask_array = np.array(mask_img, dtype=np.uint8)
            binary_mask = (mask_array > 0).astype(bool)
            
            # Compute blob statistics
            blob_stats = self._compute_blob_statistics(binary_mask)
            
            return {
                'mask': binary_mask,
                'blob_stats': blob_stats,
                'mask_file': str(mask_file),
                'processed_dir': str(processed_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load existing data for {image_name}: {e}")
            return None

    def _compute_area_change(self, init_data: Dict[str, Any], out_data: Dict[str, Any], 
                           base_name: str) -> Optional[Dict[str, Any]]:
        """
        Compute area change ratio between init and out images.
        Following the exact logic from main_extract_mistake_score_and_plot.py
        """
        try:
            init_area = init_data['blob_stats']['area']
            out_area = out_data['blob_stats']['area']
            
            # Compute area change ratio (following original logic)
            area_change_ratio = None
            if init_area > 0:
                area_change_ratio = abs(out_area - init_area) / init_area
            
            # Determine image type from base name (following original logic)
            img_type = self._img_type_from_name(base_name)
            
            return {
                'base': base_name,
                'type': img_type,
                'before_mask': init_data.get('mask_file', ''),
                'after_mask': out_data.get('mask_file', ''),
                'area_before': int(init_area),
                'area_after': int(out_area),
                'area_change': area_change_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compute area change for {base_name}: {e}")
            return None

    def _img_type_from_name(self, base: str) -> str:
        """Categorize image type from name (following original logic exactly)."""
        base_lower = base.lower()
        if 'concave_nofill' in base_lower or ('nofill' in base_lower and 'concave' not in base_lower):
            return 'concave_nofill'
        if 'concave' in base_lower:
            return 'concave'
        if 'convex' in base_lower:
            return 'convex'
        if 'no_change' in base_lower:
            return 'no_change'
        return 'unknown'

    def _perform_threshold_analysis(self, pair_details: List[Dict[str, Any]]) -> None:
        """
        Perform threshold analysis following main_extract_mistake_score_and_plot.py exactly.
        """
        self.logger.info("Performing threshold analysis (following original pipeline)")
        
        # Thresholds including 2% (following original exactly)
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06] + [i/100 for i in range(8, 21, 2)]
        
        # Image types (following original exactly)
        types = ['concave', 'concave_nofill', 'convex', 'no_change']
        
        # For each threshold, perform analysis
        for thr in thresholds:
            pct = int(round(thr * 100))
            self.logger.info(f"Analyzing threshold: {pct}%")
            
            # Create output directory for this threshold
            dir_out = os.path.join(self.threshold_results_dir, f"{pct}_comparison")
            os.makedirs(dir_out, exist_ok=True)
            
            # Save per-image details (following original exactly)
            with open(os.path.join(dir_out, 'per_image_detailed.json'), 'w') as f:
                json.dump(pair_details, f, indent=2)
            
            # Compute detections (following original logic exactly)
            detections = {t: [] for t in types}
            for detail in pair_details:
                img_type = detail['type']
                if img_type in detections:
                    area_change = detail['area_change']
                    detected = 1 if (area_change is not None and area_change > thr) else 0
                    detections[img_type].append(detected)
            
            # Overall summary (following original exactly)
            summary = {
                t: {'detected': int(sum(detections[t])), 'total': len(detections[t])} 
                for t in types
            }
            with open(os.path.join(dir_out, 'overall_comparison.json'), 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Generate plots (following original styling exactly)
            self._generate_threshold_plots(detections, types, pct, dir_out)
        
        self.logger.info(f"Completed thresholds: {[int(t*100) for t in thresholds]}%")

    def _compute_sem_binary(self, detections: np.ndarray) -> float:
        """
        Compute standard error of the mean (SEM) as percentage, then downscale by half.
        Following main_extract_mistake_score_and_plot.py exactly.
        """
        n = detections.size
        if n == 0:
            return 0.0
        p = detections.mean()
        sem = np.sqrt(p * (1 - p) / n) * 100
        return sem * 0.5  # Downscale by half

    def _generate_threshold_plots(self, detections: Dict[str, List[int]], types: List[str], 
                                pct: int, output_dir: str) -> None:
        """
        Generate plots following main_extract_mistake_score_and_plot.py styling exactly.
        """
        # Compute rates and SEMs
        rates = np.array([(np.mean(detections[t]) * 100 if detections[t] else 0.0) for t in types])
        sems = np.array([self._compute_sem_binary(np.array(detections[t])) for t in types])
        
        # Overall plot (4 categories)
        x = np.arange(len(types)) + LEFT_MARGIN
        fig, ax = plt.subplots(figsize=(4.8, 4), dpi=HIGH_DPI)
        for i, t in enumerate(types):
            ax.bar(x[i], rates[i], BAR_WIDTH,
                   color='lightgray', edgecolor='black', hatch='//',
                   yerr=sems[i], capsize=5)
        ax.set_xticks([])
        ax.set_ylabel('% Detection Rate')
        ax.set_ylim(0, 100)
        ax.set_xlim(LEFT_MARGIN - 0.8*BAR_WIDTH, LEFT_MARGIN + len(types) - 0.2*BAR_WIDTH)
        ax.set_title(f'Threshold = {pct}%')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=HIGH_DPI)
        plt.close(fig)
        
        # Three-condition plot (following original styling exactly)
        three = ['concave', 'concave_nofill', 'convex']
        colors = [
            (255/255, 188/255, 78/255),  # concave
            (209/255, 168/255, 95/255),  # concave_nofill  
            (79/255, 168/255, 78/255)    # convex
        ]
        
        # Width calculation (following original exactly: base 15% + extra 10% => 1.15 * 1.10 = 1.265)
        width_three = BAR_WIDTH * 2.1
        margin_data = BAR_WIDTH * 0.8
        x2 = np.arange(len(three)) * width_three + LEFT_MARGIN
        
        fig, ax = plt.subplots(figsize=(4.5, 6), dpi=HIGH_DPI)
        for i, t in enumerate(three):
            idx = types.index(t)
            ax.bar(x2[i], rates[idx], width_three,
                   color=colors[i], edgecolor='black', hatch='//',
                   yerr=sems[idx], capsize=5)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=TICKS_FONTSIZE)
        ax.set_ylabel('% Noticing Change', fontsize=LABEL_FONTSIZE)
        ax.set_ylim(0, 100)
        left_lim = x2[0] - width_three/2 - margin_data
        right_lim = x2[-1] + width_three/2 + margin_data
        ax.set_xlim(left_lim, right_lim)
        ax.set_title(f'Threshold = {pct}%')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'three_comparison.png'), dpi=HIGH_DPI)
        plt.close(fig)
        
        self.logger.info(f"  Generated plots for threshold {pct}%")

##############################################################################
# MAIN FUNCTION
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Change Detection Experiment - Following Original Analysis Pipeline")
    parser.add_argument("--model_interface", type=str, default="segformer",
                      choices=["segformer"], help="Model interface to use")
    parser.add_argument("--images_dir", type=str, required=False,
                      default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/detr/EXP_3_CHANGE/Data_processed/Stimuli/Exp3b_Images",
                      help="Directory containing raw image files with _init and _out pairs")
    parser.add_argument("--output_dir", type=str, required=False,
                      default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/hugging_face/model_experiments/segformer/exp3Change",
                      help="Output directory for results and processed data")
    parser.add_argument("--resume", action="store_true", default=True,
                      help="Resume processing from checkpoints (default: True)")
    parser.add_argument("--no_resume", action="store_true", default=False,
                      help="Start processing from scratch, ignoring checkpoints")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b1-finetuned-ade-512-512",
                      help="HuggingFace model repo for SegFormer")
    
    args = parser.parse_args()
    
    # Handle resume logic
    resume = args.resume and not args.no_resume
    
    # Validate inputs
    if not os.path.isdir(args.images_dir):
        print(f"Error: Images directory '{args.images_dir}' does not exist")
        sys.exit(1)
    
    # Append model suffix to output directory
    model_suffix = args.model_name.split("/")[-1]
    if not args.output_dir.endswith(model_suffix):
        args.output_dir = f"{args.output_dir}_{model_suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model interface
    if args.model_interface == "segformer":
        model_interface = SegFormerInterface(model_name=args.model_name)
    else:
        raise ValueError(f"Unknown model interface: {args.model_interface}")
    
    # Run experiment
    experiment = ChangeDetectionExperiment(model_interface, args.output_dir)
    
    try:
        experiment.run_full_experiment(
            images_dir=args.images_dir,
            resume=resume
        )
        print("Change detection experiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"Experiment failed with error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()