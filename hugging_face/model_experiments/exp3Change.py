#!/usr/bin/env python3
"""
exp3Change.py

Change Detection Experiment that processes raw image files and computes mistake scores
based on blob segmentation and change detection thresholds.
Completely self-contained from raw images to final analysis.

FIXED VERSION:
- Fixed array indexing issues by ensuring all masks are boolean
- Enhanced error handling for mask operations
- Proper type conversion for regionprops and find_contours
- Robust mask handling throughout the pipeline

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
    3. Computes change detection scores at various thresholds
    4. Generates mistake score analysis and comparison plots
    5. Saves results for each threshold comparison
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
        
        # Step 2: Process each image to extract blob information
        all_blob_data = {}
        
        for image_file in image_files:
            image_name = Path(image_file).stem
            self.logger.info(f"Processing image: {image_name}")
            
            try:
                if resume and self._is_image_already_processed(image_name):
                    self.logger.info(f"Image {image_name} already processed, loading existing data")
                    blob_data = self._load_existing_blob_data(image_name)
                else:
                    blob_data = self._process_single_image(image_file, image_name)
                
                if blob_data:
                    all_blob_data[image_name] = blob_data
                else:
                    self.logger.warning(f"No blob data extracted for image: {image_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process image {image_name}: {e}")
                continue
        
        # Step 3: Generate threshold comparisons and mistake score analysis
        if all_blob_data:
            self._analyze_threshold_comparisons(all_blob_data, thresholds)
        else:
            self.logger.warning("No blob data available - skipping analysis")
        
        self.logger.info("Change detection experiment completed successfully")

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
            
            return {
                'mask': binary_mask,
                'blob_stats': blob_stats,
                'processed_dir': str(processed_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load existing blob data for {image_name}: {e}")
            return None

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

        # Compute blob statistics
        blob_stats = self._compute_blob_statistics(chosen_mask)
        
        # Save vanilla segmentation if enabled
        if self.enable_vanilla_segmentation and self.vanilla_saver is not None:
            try:
                self.vanilla_saver.save_frame_segmentation(frame, frame_idx=0)  # Use 0 for single images
            except Exception as e:
                self.logger.warning(f"Failed to save vanilla segmentation for image {image_name}: {e}")

        return {
            'mask': chosen_mask,
            'blob_stats': blob_stats,
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

    def _analyze_threshold_comparisons(self, all_blob_data: Dict[str, Dict[str, Any]], 
                                     thresholds: List[int]) -> None:
        """Analyze different threshold comparisons and compute mistake scores."""
        self.logger.info("Analyzing threshold comparisons")
        
        # For each threshold, compute mistake scores
        for threshold in thresholds:
            self.logger.info(f"Processing threshold comparison: {threshold}")
            
            # Create output directory for this threshold
            threshold_dir = Path(self.threshold_results_dir) / f"{threshold}_comparison"
            threshold_dir.mkdir(exist_ok=True)
            
            # Compute mistake scores for this threshold
            mistake_scores = self._compute_mistake_scores(all_blob_data, threshold)
            
            # Generate plots and save results
            self._save_threshold_results(threshold_dir, threshold, mistake_scores, all_blob_data)
        
        # Generate comparative analysis across all thresholds
        self._generate_comparative_threshold_analysis(all_blob_data, thresholds)

    def _compute_mistake_scores(self, all_blob_data: Dict[str, Dict[str, Any]], 
                              threshold: int) -> Dict[str, float]:
        """
        Compute mistake scores for a given threshold.
        
        The mistake score is based on how much the detected blob deviates from 
        expected characteristics at the given threshold.
        """
        mistake_scores = {}
        
        for image_name, blob_data in all_blob_data.items():
            blob_stats = blob_data['blob_stats']
            
            # Simple mistake score: based on area deviation from threshold
            # This can be customized based on specific requirements
            area = blob_stats['area']
            expected_area = threshold * 100  # Example: threshold-based expected area
            
            if expected_area > 0:
                mistake_score = abs(area - expected_area) / expected_area
            else:
                mistake_score = 1.0  # Maximum mistake if no expected area
            
            mistake_scores[image_name] = mistake_score
            
        return mistake_scores

    def _save_threshold_results(self, threshold_dir: Path, threshold: int, 
                              mistake_scores: Dict[str, float], 
                              all_blob_data: Dict[str, Dict[str, Any]]) -> None:
        """Save results for a specific threshold comparison."""
        
        # Save mistake scores JSON
        results_json = {
            "threshold": threshold,
            "mistake_scores": mistake_scores,
            "summary": {
                "mean_mistake_score": float(np.mean(list(mistake_scores.values()))),
                "std_mistake_score": float(np.std(list(mistake_scores.values()))),
                "max_mistake_score": float(max(mistake_scores.values())),
                "min_mistake_score": float(min(mistake_scores.values())),
                "total_images": len(mistake_scores)
            }
        }
        
        with open(threshold_dir / "results.json", 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Generate mistake score distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scores = list(mistake_scores.values())
        image_names = list(mistake_scores.keys())
        
        # Bar plot of mistake scores
        bars = ax.bar(range(len(image_names)), scores, color='coral', alpha=0.7)
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Mistake Score')
        ax.set_title(f'Mistake Scores for Threshold {threshold}')
        ax.set_xticks(range(0, len(image_names), max(1, len(image_names)//10)))
        
        # Add horizontal line for mean
        mean_score = np.mean(scores)
        ax.axhline(mean_score, color='red', linestyle='--', 
                  label=f'Mean: {mean_score:.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(threshold_dir / "mistake_scores.png", dpi=300)
        plt.close(fig)
        
        # Generate histogram of mistake scores
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(scores, bins=min(20, len(scores)), color='lightblue', 
               edgecolor='black', alpha=0.7)
        ax.set_xlabel('Mistake Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Mistake Scores (Threshold {threshold})')
        ax.axvline(mean_score, color='red', linestyle='--', 
                  label=f'Mean: {mean_score:.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(threshold_dir / "mistake_distribution.png", dpi=300)
        plt.close(fig)

    def _generate_comparative_threshold_analysis(self, all_blob_data: Dict[str, Dict[str, Any]], 
                                               thresholds: List[int]) -> None:
        """Generate comparative analysis across all thresholds."""
        
        # Compute mean mistake scores for each threshold
        threshold_means = []
        threshold_stds = []
        
        for threshold in thresholds:
            mistake_scores = self._compute_mistake_scores(all_blob_data, threshold)
            scores = list(mistake_scores.values())
            threshold_means.append(np.mean(scores))
            threshold_stds.append(np.std(scores))
        
        # Plot threshold comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(thresholds, threshold_means, yerr=threshold_stds, 
                     capsize=5, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Threshold Value')
        ax.set_ylabel('Mean Mistake Score')
        ax.set_title('Mean Mistake Scores Across Different Thresholds')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, threshold_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(threshold_means)*0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        comparison_path = os.path.join(self.plots_dir, "threshold_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save summary CSV
        summary_data = []
        for i, threshold in enumerate(thresholds):
            summary_data.append({
                'threshold': threshold,
                'mean_mistake_score': threshold_means[i],
                'std_mistake_score': threshold_stds[i]
            })
        
        df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.results_dir, "threshold_summary.csv")
        df.to_csv(summary_path, index=False)
        
        self.logger.info(f"Saved comparative analysis to {comparison_path} and {summary_path}")

##############################################################################
# MAIN FUNCTION
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Change Detection Experiment - Process raw images and analyze blob segmentation")
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