 #!/usr/bin/env python3
"""
exp3Change.py

Change Detection Experiment that integrates blob segmentation and mistake score analysis.
Uses a configurable model interface for object detection and segmentation.

Usage:
    python exp3Change.py --model_interface segformer --images_folder /path/to/images --output_dir /path/to/output
"""

import os
import re
import argparse
import json
from datetime import datetime
from typing import List, Tuple, Dict, Any
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours

# Import model interfaces and video processor
from segformer.segformer_interface import SegFormerInterface, ModelInterface
from video_processor import VideoProcessor


##############################################################################
# EXPERIMENT CLASS
##############################################################################

class ChangeDetectionExperiment:
    """
    Experiment 3: Change Detection Analysis
    
    This experiment:
    1. Processes single images using a model interface to segment blobs
    2. Compares before/after image pairs to detect changes
    3. Analyzes area-change thresholds across different conditions
    4. Generates detection rate plots and statistics
    """
    
    def __init__(self, model_interface: ModelInterface, output_dir: str, logger: logging.Logger = None):
        self.model_interface = model_interface
        self.output_dir = output_dir
        self.logger = logger or self._setup_logger()
        
        # Initialize video processor (though Change experiment primarily processes images)
        self.video_processor = VideoProcessor(model_interface, self.logger)
        
        # Create output subdirectories
        self.processed_dir = os.path.join(output_dir, "processed_images")
        self.results_dir = os.path.join(output_dir, "threshold_results")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.processed_videos_dir = os.path.join(output_dir, "processed_videos")
        
        for dir_path in [self.processed_dir, self.results_dir, self.plots_dir, self.logs_dir, self.processed_videos_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        self.logger.info(f"Initialized Change Detection Experiment with output dir: {output_dir}")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(self.logs_dir, f"change_exp_{timestamp}.log")

        logger = logging.getLogger(f"change_exp_{timestamp}")
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

    def run_full_experiment(self, images_folder: str, resume: bool = True) -> None:
        """Run the complete change detection experiment."""
        self.logger.info("Starting full change detection experiment")
        
        # Step 1: Process images to generate segmentation masks
        self._segment_images(images_folder, resume=resume)
        
        # Step 2: Analyze threshold-based change detection
        self._analyze_change_thresholds()
        
        self.logger.info("Change detection experiment completed successfully")

    def _segment_images(self, images_folder: str, resume: bool = True) -> None:
        """Process all images to generate segmentation masks and visualizations."""
        self.logger.info(f"Starting image segmentation from: {images_folder}")
        
        # Load model if not already loaded
        if not hasattr(self.model_interface, 'model') or self.model_interface.model is None:
            self.logger.info("Loading model...")
            self.model_interface.load_model()
        
        # Get list of image files
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        image_files = [
            os.path.join(images_folder, f) for f in sorted(os.listdir(images_folder))
            if os.path.splitext(f)[1].lower() in exts
        ]
        
        if not image_files:
            self.logger.error("No images found in folder. Exiting.")
            return
            
        self.logger.info(f"Found {len(image_files)} images to process")
        
        for img_path in image_files:
            try:
                # Check if already processed when resume=True
                if resume and self._is_image_already_processed(img_path):
                    self.logger.info(f"Skipping already processed image: {os.path.basename(img_path)}")
                    continue
                    
                self._process_single_image(img_path)
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")

    def _process_single_image(self, img_path: str) -> None:
        """Process a single image to generate segmentation and outputs."""
        name = os.path.splitext(os.path.basename(img_path))[0]
        self.logger.info(f"Processing image: {name}")
        
        # Setup output directories for this image
        model_prefix = "segformer_model"  # Can be made configurable
        base_output = os.path.join(self.processed_dir, f"{model_prefix}_{name}")
        
        dirs = {
            "blobs": os.path.join(base_output, "frames_blobs"),
            "collage": os.path.join(base_output, "frames_collage"),
            "mask": os.path.join(base_output, "frames_masks_nonmem"),
            "proc": os.path.join(base_output, "frames_processed"),
        }
        
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
        
        # Load and preprocess image
        frame = np.array(Image.open(img_path).convert('RGB'))
        H, W, _ = frame.shape
        self.logger.debug(f"Image shape: {H}x{W} RGB")
        
        # Detect blob using simple thresholding (could be enhanced)
        blob = self._detect_blob(frame)
        if blob is None:
            self.logger.warning(f"No blob found in {name} - skipping")
            return
        
        # Save blob overlay
        overlay = frame.copy()
        overlay[blob] = [255, 0, 0]
        blob_file = os.path.join(dirs['blobs'], f"{name}_blobs.png")
        Image.fromarray(overlay).save(blob_file)
        self.logger.debug(f"Saved blob overlay: {blob_file}")
        
        # Run model inference
        pil_image = Image.fromarray(frame)
        predictions = self.model_interface.infer_image(pil_image)
        
        # Extract candidate masks from model predictions
        candidates = self._extract_candidate_masks(predictions, H, W)
        self.logger.debug(f"Generated {len(candidates)} candidate masks")
        
        # Choose best mask based on IoU with detected blob
        best_mask = self._choose_best_mask(blob, candidates)
        
        # Save chosen mask
        if best_mask is not None and best_mask.sum() > 0:
            mask_file = os.path.join(dirs['mask'], f"mask_{name}.png")
            Image.fromarray((best_mask.astype(np.uint8) * 255)).save(mask_file)
            self.logger.debug(f"Saved chosen mask: {mask_file}")
        else:
            self.logger.warning(f"No suitable mask found for {name}")
            # Create empty mask file for consistency
            mask_file = os.path.join(dirs['mask'], f"mask_{name}.png")
            Image.fromarray(np.zeros((H, W), dtype=np.uint8)).save(mask_file)
        
        # Generate collage of top candidates
        self._generate_candidate_collage(frame, blob, candidates, dirs['collage'], name)
        
        # Generate final overlay with polygon
        self._generate_final_overlay(frame, best_mask, dirs['proc'], name, H, W)

    def _detect_blob(self, frame: np.ndarray, thresholds: Tuple[int, ...] = (30, 15, 5)) -> np.ndarray:
        """Detect the main blob in the image using intensity thresholding."""
        gray = frame.sum(axis=2)
        
        for thr in thresholds:
            labeled = label(gray > thr, connectivity=2)
            regions = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)
            
            if regions:
                self.logger.debug(f"Blob detected with threshold {thr} (area={regions[0].area})")
                return labeled == regions[0].label
            else:
                self.logger.debug(f"No blob found at threshold {thr}")
        
        return None

    def _extract_candidate_masks(self, predictions: Dict[str, Any], height: int, width: int) -> List[np.ndarray]:
        """Extract candidate masks from model predictions."""
        pred_masks = predictions['pred_masks']  # Shape: (1, num_queries, H, W)
        
        candidates = []
        for i in range(pred_masks.shape[1]):
            mask = pred_masks[0, i].cpu().numpy()  # (H, W)
            
            # Resize if needed
            if mask.shape != (height, width):
                mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
                mask_resized = F.interpolate(mask_tensor, size=(height, width), mode='bilinear', align_corners=False)
                mask = mask_resized.squeeze().numpy()
            
            # Threshold to binary
            binary_mask = mask > 0.5
            
            # Split into connected components
            labeled = label(binary_mask, connectivity=2)
            for lbl in range(1, labeled.max() + 1):
                component_mask = (labeled == lbl)
                candidates.append(component_mask)
        
        return candidates

    def _choose_best_mask(self, blob: np.ndarray, candidates: List[np.ndarray]) -> np.ndarray:
        """Choose the best mask based on IoU with the detected blob."""
        if not candidates:
            return None
        
        best_iou = -1
        best_mask = None
        
        for candidate in candidates:
            iou_val = self._compute_iou(blob, candidate)
            if iou_val > best_iou:
                best_iou = iou_val
                best_mask = candidate
        
        return best_mask

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return 0.0 if union == 0 else intersection / union

    def _generate_candidate_collage(self, frame: np.ndarray, blob: np.ndarray, 
                                  candidates: List[np.ndarray], output_dir: str, name: str) -> None:
        """Generate a collage showing top candidate masks."""
        if not candidates:
            return
        
        # Compute IoU scores and get top 10
        ious = [-self._compute_iou(blob, mask) for mask in candidates]
        best_indices = np.argsort(ious)[:10]
        
        # Create collage
        fig, axes = plt.subplots(1, 10, figsize=(25, 3), dpi=100)
        if len(best_indices) < 10:
            # Hide unused subplots
            for i in range(len(best_indices), 10):
                axes[i].axis('off')
        
        for i, idx in enumerate(best_indices):
            if i >= 10:
                break
                
            overlay = frame.copy()
            overlay[blob] = [0, 255, 0]  # Green for blob
            overlay[candidates[idx]] = [255, 0, 0]  # Red for candidate
            
            axes[i].imshow(overlay)
            axes[i].set_title(f"#{idx}\n{ious[idx]:.3f}", fontsize=6)
            axes[i].axis('off')
        
        collage_file = os.path.join(output_dir, f"{name}_collage.png")
        fig.savefig(collage_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        self.logger.debug(f"Saved collage: {collage_file}")

    def _generate_final_overlay(self, frame: np.ndarray, mask: np.ndarray, 
                              output_dir: str, name: str, height: int, width: int) -> None:
        """Generate final overlay with polygon contour."""
        final_image = Image.fromarray(frame.copy())
        
        if mask is not None and mask.sum() > 0:
            # Find contour and create polygon
            contours = find_contours(mask.astype(np.uint8), 0.5)
            if contours:
                # Use the largest contour
                largest_contour = max(contours, key=len)
                
                # Convert to PIL format (note: contours are in (row, col) format)
                polygon_points = [(int(point[1]), int(point[0])) for point in largest_contour]
                
                # Draw polygon overlay
                draw = ImageDraw.Draw(final_image, 'RGBA')
                if len(polygon_points) > 2:
                    draw.polygon(polygon_points, fill=(255, 0, 0, 120))
                    
                    # Add centroid label
                    centroid_x = sum(p[0] for p in polygon_points) / len(polygon_points)
                    centroid_y = sum(p[1] for p in polygon_points) / len(polygon_points)
                    draw.text((centroid_x, centroid_y), 'Blob', fill=(255, 255, 255, 255))
        
        final_file = os.path.join(output_dir, f"{name}_overlay.png")
        final_image.save(final_file)
        self.logger.debug(f"Saved final overlay: {final_file}")

    def _analyze_change_thresholds(self) -> None:
        """Analyze area-change thresholds across image pairs."""
        self.logger.info("Starting threshold analysis for change detection")
        
        # Find before/after image pairs
        pairs = self._find_image_pairs()
        if not pairs:
            self.logger.error("No valid image pairs found for analysis")
            return
        
        self.logger.info(f"Found {len(pairs)} image pairs for analysis")
        
        # Define thresholds including 2%
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06] + [i/100 for i in range(8, 21, 2)]
        
        # Compute per-image changes
        change_details = self._compute_change_details(pairs)
        
        # Analyze each threshold
        for threshold in thresholds:
            self._analyze_single_threshold(threshold, change_details)

    def _find_image_pairs(self) -> List[Tuple[str, str, str]]:
        """Find before/after image pairs in the processed directory."""
        pairs = []
        
        # Look for directories ending with '_init' and find corresponding '_out'
        for item_name in sorted(os.listdir(self.processed_dir)):
            if not item_name.endswith('_init'):
                continue
            if 'catch_shape' in item_name:
                continue
                
            base_name = item_name[:-5]  # Remove '_init'
            init_dir = os.path.join(self.processed_dir, item_name)
            out_dir = os.path.join(self.processed_dir, base_name + '_out')
            
            if os.path.isdir(init_dir) and os.path.isdir(out_dir):
                pairs.append((base_name, init_dir, out_dir))
                self.logger.debug(f"Found image pair: {base_name}")
        
        return pairs

    def _compute_change_details(self, pairs: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Compute area change details for all image pairs."""
        details = []
        
        for base_name, init_dir, out_dir in pairs:
            try:
                # Load masks
                init_mask = self._load_mask_from_dir(os.path.join(init_dir, 'frames_masks_nonmem'))
                out_mask = self._load_mask_from_dir(os.path.join(out_dir, 'frames_masks_nonmem'))
                
                # Compute areas
                area_before = init_mask.sum()
                area_after = out_mask.sum()
                
                # Compute relative change
                if area_before == 0:
                    area_change_ratio = None
                else:
                    area_change_ratio = abs(area_after - area_before) / area_before
                
                details.append({
                    'base': base_name,
                    'type': self._classify_image_type(base_name),
                    'before_mask': os.path.join(init_dir, 'frames_masks_nonmem'),
                    'after_mask': os.path.join(out_dir, 'frames_masks_nonmem'),
                    'area_before': int(area_before),
                    'area_after': int(area_after),
                    'area_change': area_change_ratio
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process pair {base_name}: {e}")
                continue
        
        return details

    def _load_mask_from_dir(self, mask_dir: str) -> np.ndarray:
        """Load the first PNG mask from a directory."""
        for filename in os.listdir(mask_dir):
            if filename.lower().endswith('.png'):
                mask_path = os.path.join(mask_dir, filename)
                img = Image.open(mask_path).convert('L')
                return np.array(img) > 0
        
        raise FileNotFoundError(f"No PNG mask found in {mask_dir}")

    def _classify_image_type(self, base_name: str) -> str:
        """Classify image type based on name."""
        if 'concave_nofill' in base_name or ('nofill' in base_name and 'concave' not in base_name):
            return 'concave_nofill'
        elif 'concave' in base_name:
            return 'concave'
        elif 'convex' in base_name:
            return 'convex'
        elif 'no_change' in base_name:
            return 'no_change'
        else:
            return 'unknown'

    def _analyze_single_threshold(self, threshold: float, change_details: List[Dict[str, Any]]) -> None:
        """Analyze detection rates for a single threshold."""
        pct = int(round(threshold * 100))
        threshold_dir = os.path.join(self.results_dir, f"{pct}_comparison")
        os.makedirs(threshold_dir, exist_ok=True)
        
        # Save per-image details
        details_file = os.path.join(threshold_dir, 'per_image_detailed.json')
        with open(details_file, 'w') as f:
            json.dump(change_details, f, indent=2)
        
        # Compute detection rates by type
        types = ['concave', 'concave_nofill', 'convex', 'no_change']
        detections = {t: [] for t in types}
        
        for detail in change_details:
            img_type = detail['type']
            if img_type in detections:
                change_ratio = detail['area_change']
                detected = 1 if (change_ratio is not None and change_ratio > threshold) else 0
                detections[img_type].append(detected)
        
        # Compute summary statistics
        summary = {}
        for img_type in types:
            det_list = detections[img_type]
            summary[img_type] = {
                'detected': int(sum(det_list)),
                'total': len(det_list),
                'rate': (sum(det_list) / len(det_list) * 100) if det_list else 0.0
            }
        
        # Save summary
        summary_file = os.path.join(threshold_dir, 'overall_comparison.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate plots
        self._generate_threshold_plots(threshold_dir, summary, types, pct)
        
        self.logger.info(f"Completed threshold analysis for {pct}%")

    def _generate_threshold_plots(self, output_dir: str, summary: Dict[str, Dict], 
                                 types: List[str], pct: int) -> None:
        """Generate detection rate plots for a threshold."""
        
        # Plot parameters
        label_fontsize = 21
        ticks_fontsize = 19
        width = 1.0
        left_margin = 0.5
        high_dpi = 200
        
        # Compute rates and error bars
        rates = np.array([summary[t]['rate'] for t in types])
        sems = np.array([self._compute_sem_binary(summary[t]) for t in types])
        
        # Overall plot (all 4 conditions)
        x = np.arange(len(types)) + left_margin
        fig, ax = plt.subplots(figsize=(4.8, 4), dpi=high_dpi)
        
        for i, img_type in enumerate(types):
            ax.bar(x[i], rates[i], width,
                   color='lightgray', edgecolor='black', hatch='//',
                   yerr=sems[i], capsize=5)
        
        ax.set_xticks([])
        ax.set_ylabel('% Detection Rate')
        ax.set_ylim(0, 100)
        ax.set_xlim(left_margin - 0.8*width, left_margin + len(types) - 0.2*width)
        ax.set_title(f'Threshold = {pct}%')
        plt.tight_layout()
        
        overall_plot = os.path.join(output_dir, 'overall_comparison.png')
        fig.savefig(overall_plot, dpi=high_dpi)
        plt.close(fig)
        
        # Three-condition plot (excluding no_change)
        three_types = ['concave', 'concave_nofill', 'convex']
        three_colors = [
            (255/255, 188/255, 78/255),  # concave
            (209/255, 168/255, 95/255),  # concave_nofill  
            (79/255, 168/255, 78/255)    # convex
        ]
        
        width_three = width * 2.1
        margin_data = width * 0.8
        x2 = np.arange(len(three_types)) * width_three + left_margin
        
        fig, ax = plt.subplots(figsize=(4.5, 6), dpi=high_dpi)
        
        for i, img_type in enumerate(three_types):
            type_idx = types.index(img_type)
            ax.bar(x2[i], rates[type_idx], width_three,
                   color=three_colors[i], edgecolor='black', hatch='//',
                   yerr=sems[type_idx], capsize=5)
        
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=ticks_fontsize)
        ax.set_ylabel('% Noticing Change', fontsize=label_fontsize)
        ax.set_ylim(0, 100)
        
        left_lim = x2[0] - width_three/2 - margin_data
        right_lim = x2[-1] + width_three/2 + margin_data
        ax.set_xlim(left_lim, right_lim)
        ax.set_title(f'Threshold = {pct}%')
        plt.tight_layout()
        
        three_plot = os.path.join(output_dir, 'three_comparison.png')
        fig.savefig(three_plot, dpi=high_dpi)
        plt.close(fig)

    def _is_image_already_processed(self, img_path: str) -> bool:
        """Check if an image has already been processed."""
        name = os.path.splitext(os.path.basename(img_path))[0]
        model_prefix = "segformer_model"
        base_output = os.path.join(self.processed_dir, f"{model_prefix}_{name}")
        
        # Check if the key output directories and files exist
        mask_dir = os.path.join(base_output, "frames_masks_nonmem")
        proc_dir = os.path.join(base_output, "frames_processed")
        
        if not (os.path.exists(mask_dir) and os.path.exists(proc_dir)):
            return False
        
        # Check if mask file exists
        mask_file = os.path.join(mask_dir, f"mask_{name}.png")
        if not os.path.exists(mask_file):
            return False
        
        # Check if overlay file exists
        overlay_file = os.path.join(proc_dir, f"{name}_overlay.png")
        return os.path.exists(overlay_file)

    def _compute_sem_binary(self, stats_dict: Dict[str, Any]) -> float:
        """Compute standard error of the mean for binary detection data."""
        total = stats_dict['total']
        if total == 0:
            return 0.0
        
        rate = stats_dict['rate'] / 100.0  # Convert percentage to proportion
        sem = np.sqrt(rate * (1 - rate) / total) * 100  # Convert back to percentage
        return sem * 0.5  # Downscale by half as in original


##############################################################################
# MAIN FUNCTION
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Run Change Detection Experiment with configurable model interface")
    
    parser.add_argument("--model_interface", default="segformer", 
                       choices=["segformer"], 
                       help="Model interface to use")
    parser.add_argument("--images_folder", required=True,
                       help="Folder containing input images")
    parser.add_argument("--output_dir", required=True,
                       help="Directory for experiment outputs")
    parser.add_argument("--model_name", default="nvidia/segformer-b5-finetuned-ade-640-640",
                       help="Model name/path for the interface")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume from previous processing if possible")
    parser.add_argument("--no_resume", action="store_true",
                       help="Force restart from beginning")
    
    args = parser.parse_args()
    
    # Create model interface
    if args.model_interface == "segformer":
        model_interface = SegFormerInterface(model_name=args.model_name)
    else:
        raise ValueError(f"Unknown model interface: {args.model_interface}")
    
    # Create and run experiment
    experiment = ChangeDetectionExperiment(model_interface, args.output_dir)
    
    # Determine resume flag
    resume = args.resume and not args.no_resume
    
    experiment.run_full_experiment(args.images_folder, resume=resume)


if __name__ == "__main__":
    main()