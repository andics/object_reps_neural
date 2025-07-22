#!/usr/bin/env python3
"""
video_processor.py

Comprehensive video processing class that handles video frame extraction,
model inference, mask generation, and output creation. 
Closely follows the structure and logic of main_gen_vids_and_meshes.py
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import deque
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, find_contours
from scipy.optimize import linear_sum_assignment

from segformer.segformer_interface import ModelInterface
from vanilla_segmentation import VanillaSegmentationSaver


class VideoProcessor:
    """
    Comprehensive video processor that handles:
    - Video frame extraction and processing
    - Model inference using configurable interfaces
    - Blob detection and mask assignment with state tracking
    - Memory-based mask tracking with configurable strategies
    - Collage generation showing mask fitting quality
    - Output generation (masks, visualizations, videos)
    
    Args:
        model_interface: The model interface to use for inference
        n_blobs: Number of blobs to detect and track (default: 2)
        logger: Optional logger instance for logging messages
        blob_1_memory_strategy: Strategy for blob 1 memory updates ('exponential' or 'running_average')
        blob_1_running_avg_window: Window size for running average (default: 10)
        blob_1_memory_freeze_frame: Frame after which blob 1 memory stops updating (default: None)
    """
    
    def __init__(self, model_interface: ModelInterface, n_blobs: int = 2, logger: logging.Logger = None,
                 blob_1_memory_strategy: str = 'exponential', blob_1_running_avg_window: int = 10,
                 blob_1_memory_freeze_frame: Optional[int] = None, enable_vanilla_segmentation: bool = True):
        self.model_interface = model_interface
        self.logger = logger or logging.getLogger(__name__)
        
        # Processing parameters
        self.n_blobs = n_blobs
        self.initial_skip_frames = 13
        self.alpha = 0.7  # Memory decay factor for exponential averaging
        self.black_thresh = 30  # Threshold for blob detection
        
        # Memory update strategy parameters
        self.blob_1_memory_strategy = blob_1_memory_strategy
        self.blob_1_running_avg_window = blob_1_running_avg_window
        self.blob_1_memory_freeze_frame = blob_1_memory_freeze_frame
        
        # Vanilla segmentation option
        self.enable_vanilla_segmentation = enable_vanilla_segmentation
        self.vanilla_saver = None
        
        # Memory for tracking masks across frames
        self.mem_floats = None
        self.current_video_shape = None
        
        # Running average storage for blob 1
        self.blob_1_mask_history = deque(maxlen=blob_1_running_avg_window)
        
        # Blob state tracking to prevent false detections
        self.blob_1_disappeared = False
        self.blob_1_disappeared_frame = None
        self.blob_1_missing_count = 0
        self.blob_1_missing_threshold = 5  # Frames of absence before considering disappeared
        
        # Track frozen memory masks for reuse
        self.frozen_memory_masks = {}  # blob_idx -> frozen mask file path
        
        self.logger.info(f"VideoProcessor initialized with {self.n_blobs} blobs to detect")
        self.logger.info(f"Blob 1 memory strategy: {self.blob_1_memory_strategy}")
        if self.blob_1_memory_strategy == 'running_average':
            self.logger.info(f"Blob 1 running average window: {self.blob_1_running_avg_window}")
        if self.blob_1_memory_freeze_frame is not None:
            self.logger.info(f"Blob 1 memory will freeze after frame: {self.blob_1_memory_freeze_frame}")
        if self.enable_vanilla_segmentation:
            self.logger.info("Vanilla segmentation enabled")
        
    def setup_output_directories(self, video_path: str, output_root: str, model_prefix: str = None) -> Dict[str, str]:
        """Setup organized output directory structure for video processing."""
        
        if model_prefix is None:
            model_prefix = "segformer_model"
            
        video_prefix = self._parse_video_prefix(video_path)
        
        # Create root folder structure (matching main_gen_vids_and_meshes.py)
        root_folder = os.path.join(output_root, f"{model_prefix}-{video_prefix}")
        
        directories = {
            'root': root_folder,
            'frames_blobs': os.path.join(root_folder, "frames_blobs"),
            'frames_masks': os.path.join(root_folder, "frames_masks"),
            'frames_masks_nonmem': os.path.join(root_folder, "frames_masks_nonmem"),
            'frames_json_memory': os.path.join(root_folder, "frames_json_memory_processed"),
            'frames_collage': os.path.join(root_folder, "frames_collage"),
            'frames_memory_collage': os.path.join(root_folder, "frames_memorycollage"),
            'frames_processed': os.path.join(root_folder, "frames_processed"),
            'videos_processed': os.path.join(root_folder, "videos_processed"),
            'metadata': os.path.join(root_folder, "metadata"),
            'org_segmentation': os.path.join(root_folder, "org_segmentation")
        }
        
        # Create all directories
        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize vanilla segmentation saver if enabled
        if self.enable_vanilla_segmentation:
            self.vanilla_saver = VanillaSegmentationSaver(
                model_interface=self.model_interface,
                output_dir=directories['org_segmentation'],
                logger=self.logger
            )
            
        return directories
    
    def check_processing_status(self, directories: Dict[str, str], total_frames: int = None) -> Dict[str, Any]:
        """Check which processing steps have been completed."""
        status = {
            'frames_extracted': False,
            'masks_generated': False,
            'video_created': False,
            'last_processed_frame': -1,
            'total_frames': total_frames,
            'can_resume': False,
            'processing_complete': False
        }
        
        # Check if metadata exists
        metadata_file = os.path.join(directories['metadata'], 'processing_status.json')
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    saved_status = json.load(f)
                status.update(saved_status)
            except Exception as e:
                self.logger.warning(f"Could not load processing status: {e}")
        
        # Check actual files to verify status
        processed_frames = self._count_processed_frames(directories['frames_processed'])
        status['last_processed_frame'] = processed_frames - 1
        status['frames_extracted'] = processed_frames > 0
        
        # Check if final video exists
        video_files = list(Path(directories['videos_processed']).glob("*.mp4"))
        status['video_created'] = len(video_files) > 0
        
        # Fix resume logic
        if total_frames:
            if processed_frames == 0:
                # No processing done yet
                status['can_resume'] = False
                status['processing_complete'] = False
            elif processed_frames >= total_frames:
                # Processing is complete
                status['can_resume'] = False
                status['processing_complete'] = True
            else:
                # Partial processing - can resume
                status['can_resume'] = True
                status['processing_complete'] = False
        
        return status
    
    def _save_processing_status(self, directories: Dict[str, str], status: Dict[str, Any]) -> None:
        """Save current processing status to JSON file."""
        try:
            metadata_file = os.path.join(directories['metadata'], 'processing_status.json')
            with open(metadata_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save processing status: {e}")
    
    def process_video(self, video_path: str, output_root: str, model_prefix: str = None, 
                     resume: bool = True) -> Dict[str, str]:
        """Process a complete video through the full pipeline."""
        self.logger.info(f"Starting video processing: {video_path}")
        
        # Reset blob state tracking for new video
        self._reset_blob_state()
        
        # Setup directories
        directories = self.setup_output_directories(video_path, output_root, model_prefix)
        
        # Load model if not already loaded
        if not hasattr(self.model_interface, 'model') or self.model_interface.model is None:
            self.logger.info("Loading model...")
            self.model_interface.load_model()
        
        # Get video metadata
        video_metadata = self._get_video_metadata(video_path)
        self.current_video_shape = (video_metadata['height'], video_metadata['width'])
        
        # Check processing status
        status = self.check_processing_status(directories, video_metadata['total_frames'])
        
        # Handle different processing scenarios
        if status['processing_complete'] and not resume:
            self.logger.info("Processing already complete, but not resuming. Starting from beginning.")
            start_frame = 0
            self._initialize_memory(video_metadata['height'], video_metadata['width'])
        elif status['processing_complete'] and resume:
            self.logger.info("Processing already complete. Checking video creation...")
            if not status['video_created']:
                # Try to create video from existing frames
                if self._attempt_video_generation_from_existing_frames(directories, video_metadata):
                    self.logger.info("Video created from existing frames")
                    status['video_created'] = True
                    self._save_processing_status(directories, status)
                else:
                    self.logger.warning("Could not create video from existing frames, will reprocess")
                    # Fall through to reprocessing
                    start_frame = 0
                    self._initialize_memory(video_metadata['height'], video_metadata['width'])
            else:
                self.logger.info(f"Video processing already completed: {directories['root']}")
                return directories
        elif resume and status['can_resume']:
            self.logger.info(f"Partial processing detected. Last frame: {status['last_processed_frame']}")
            
            # Check if we can create video from existing frames
            if self._attempt_video_generation_from_existing_frames(directories, video_metadata):
                self.logger.info("Video created from existing partial frames")
                status['video_created'] = True
                status['processing_complete'] = True
                self._save_processing_status(directories, status)
                return directories
            else:
                # Not enough frames, continue processing
                self.logger.info(f"Not enough frames for video, resuming processing from frame {status['last_processed_frame'] + 1}")
                start_frame = status['last_processed_frame'] + 1
                self._initialize_memory(video_metadata['height'], video_metadata['width'])
        else:
            self.logger.info("Starting processing from beginning")
            start_frame = 0
            self._initialize_memory(video_metadata['height'], video_metadata['width'])
        
        # Process frames
        self._process_video_frames(
            video_path, directories, video_metadata, start_frame
        )
        
        # Update status after frame processing
        status = self.check_processing_status(directories, video_metadata['total_frames'])
        status['frames_extracted'] = True
        status['masks_generated'] = True
        self._save_processing_status(directories, status)
        
        # ALWAYS create final video if it doesn't exist
        videos_dir = directories['videos_processed']
        video_files_exist = any(f.endswith(('.mp4', '.avi', '.mov')) for f in os.listdir(videos_dir) if os.path.isfile(os.path.join(videos_dir, f)))
        
        if not video_files_exist:
            self.logger.info("No video file found, creating final video...")
            self._create_final_video(directories, video_metadata)
            status['video_created'] = True
            status['processing_complete'] = True
            self._save_processing_status(directories, status)
        else:
            self.logger.info("Video file already exists")
            status['video_created'] = True
        
        self.logger.info(f"Video processing completed: {directories['root']}")
        return directories
    
    def _reset_blob_state(self) -> None:
        """Reset blob state tracking for new video processing."""
        self.blob_1_disappeared = False
        self.blob_1_disappeared_frame = None
        self.blob_1_missing_count = 0
        self.blob_1_mask_history.clear()
        self.frozen_memory_masks.clear()
        self.logger.info("Reset blob state tracking for new video")
    
    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract metadata from video file."""
        try:
            reader = imageio.get_reader(video_path, format='ffmpeg')
            meta = reader.get_meta_data()
            
            # Get first frame to determine dimensions
            first_frame = reader.get_data(0)
            height, width = first_frame.shape[:2]
            
            # Count total frames efficiently
            try:
                frame_count = reader.count_frames()
            except:
                # Fallback: count manually
                frame_count = 0
                try:
                    while True:
                        reader.get_data(frame_count)
                        frame_count += 1
                except IndexError:
                    pass
            
            reader.close()
            
            metadata = {
                'width': width,
                'height': height,
                'fps': float(meta.get('fps', 30)),
                'total_frames': frame_count,
                'duration': frame_count / float(meta.get('fps', 30))
            }
            
            self.logger.info(f"Video metadata: {metadata}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to read video metadata: {e}")
            raise
    
    def _initialize_memory(self, height: int, width: int) -> None:
        """Initialize memory arrays for mask tracking."""
        self.mem_floats = [
            np.zeros((height, width), dtype=np.float32) 
            for _ in range(self.n_blobs)
        ]
    
    def _process_video_frames(self, video_path: str, directories: Dict[str, str], 
                            video_metadata: Dict[str, Any], start_frame: int = 0) -> None:
        """Process all frames in the video."""
        
        reader = imageio.get_reader(video_path, format='ffmpeg')
        flip_blobs = self._video_is_flipped(video_path)
        
        H, W = video_metadata['height'], video_metadata['width']
        total_frames = video_metadata['total_frames']
        
        try:
            for frame_idx in range(start_frame, total_frames):
                try:
                    frame = reader.get_data(frame_idx)
                except IndexError:
                    break
                
                # Ensure frame has correct dimensions
                frame = self._normalize_frame_dimensions(frame, H, W)
                
                # Process frame
                self._process_single_frame(
                    frame, frame_idx, directories, flip_blobs, H, W
                )
                
                # Save progress periodically (every 50 frames) and update status
                if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
                    processed_frames = frame_idx + 1
                    status = {
                        'frames_extracted': True,
                        'masks_generated': processed_frames > start_frame,
                        'video_created': False,
                        'last_processed_frame': frame_idx,
                        'total_frames': total_frames,
                        'can_resume': processed_frames < total_frames,
                        'processing_complete': processed_frames >= total_frames
                    }
                    self._save_processing_status(directories, status)
                
                # Log progress
                if frame_idx % 10 == 0:
                    self.logger.info(f"Processed frame {frame_idx}/{total_frames}")
        
        finally:
            reader.close()
    
    def _process_single_frame(self, frame: np.ndarray, frame_idx: int, 
                            directories: Dict[str, str], flip_blobs: bool, H: int, W: int) -> None:
        """Process a single video frame following main_gen_vids_and_meshes.py logic."""
        
        # Track current frame index for frozen mask logic
        self.current_frame_idx = frame_idx
        
        # Handle initial frames (still process them, but no blob detection)
        if frame_idx < self.initial_skip_frames:
            # Still run model inference and save vanilla segmentation for initial frames
            if self.enable_vanilla_segmentation and self.vanilla_saver is not None:
                try:
                    self.vanilla_saver.save_frame_segmentation(frame, frame_idx)
                except Exception as e:
                    self.logger.warning(f"Failed to save vanilla segmentation for frame {frame_idx}: {e}")
            
            # Save the original frame as processed frame
            output_path = os.path.join(directories['frames_processed'], f"frame_{frame_idx:06d}.png")
            Image.fromarray(frame).save(output_path)
            
            # Create empty memory JSON
            empty_data = {}
            mem_json_path = os.path.join(directories['frames_json_memory'], f"frame_{frame_idx:06d}.json")
            with open(mem_json_path, 'w') as f:
                json.dump(empty_data, f, indent=2)
            return
        
        try:
            # 1. Detect color blobs (ground truth) with state tracking
            blob_masks = self._find_color_blobs(frame, flip_blobs, frame_idx)
            
            # 2. ALWAYS run model inference to get predicted masks (even if no blobs detected)
            pred_masks = self._run_model_inference_with_splitting(frame, H, W)
        except Exception as e:
            import traceback
            self.logger.error(f"Failed blob detection or model inference for frame {frame_idx}: {e}")
            self.logger.error("Full traceback:")
            for line in traceback.format_exc().splitlines():
                self.logger.error(line)
            # Use empty masks and continue
            blob_masks = []
            pred_masks = []
        
        # 3. Save vanilla segmentation FIRST (before any blob-specific processing)
        if self.enable_vanilla_segmentation and self.vanilla_saver is not None:
            try:
                self.vanilla_saver.save_frame_segmentation(frame, frame_idx)
            except Exception as e:
                import traceback
                self.logger.error(f"Failed to save vanilla segmentation for frame {frame_idx}: {e}")
                self.logger.error("Full traceback:")
                for line in traceback.format_exc().splitlines():
                    self.logger.error(line)
        
        # 4. Handle case where no color blobs detected - CONTINUE processing with empty masks
        if len(blob_masks) == 0:
            self.logger.debug(f"Frame {frame_idx}: No color blobs detected, continuing with empty mask processing")
            # Create empty blob masks for consistency
            blob_masks = []
            assigned_masks = []
        else:
            try:
                # 5. Save blob visualization (only if blobs detected)
                self._save_blob_visualization(frame, blob_masks, frame_idx, directories)
                
                # 6. Assign masks to blobs using bipartite matching
                assigned_indices, cost_matrix = self._bipartite_assign_blobs_to_masks(blob_masks, pred_masks)
                
                # 7. Create collage showing mask fitting quality
                if cost_matrix is not None and frame_idx >= 30:
                    self._create_and_save_collage(frame, blob_masks, pred_masks, cost_matrix, frame_idx, directories)
                
                # 8. Get assigned masks
                assigned_masks = []
                for blob_idx in range(len(blob_masks)):
                    pred_idx = assigned_indices[blob_idx]
                    if pred_idx is not None:
                        assigned_masks.append(pred_masks[pred_idx])
                    else:
                        assigned_masks.append(None)
                        
            except Exception as e:
                import traceback
                self.logger.error(f"Failed blob assignment processing for frame {frame_idx}: {e}")
                self.logger.error("Full traceback:")
                for line in traceback.format_exc().splitlines():
                    self.logger.error(line)
                # Continue with empty masks
                assigned_masks = []
        
        try:
            # 9. Save non-memory masks (even if empty)
            self._save_nonmemory_masks(assigned_masks, frame_idx, directories)
            
            # 10. Update memory with custom strategy (CRITICAL: Always update memory)
            self._update_memory_masks_with_strategy(assigned_masks, frame_idx)
            
            # 11. Get memory masks (CRITICAL: Always get memory masks)
            memory_masks = self._get_memory_masks()
            
            # 12. Save memory masks (CRITICAL: Always save memory masks with freeze frame logic)
            self._save_memory_masks_with_freeze_logic(memory_masks, frame_idx, directories)
            
            # 13. Create memory collage (even if empty)
            self._create_memory_collage(frame, assigned_masks, memory_masks, frame_idx, directories)
            
            # 14. Create final overlay and save (CRITICAL: ALWAYS save final frame)
            self._create_and_save_final_overlay(frame, memory_masks, frame_idx, directories, flip_blobs, H, W)
            
        except Exception as e:
            import traceback
            self.logger.error(f"CRITICAL ERROR in memory operations for frame {frame_idx}: {e}")
            self.logger.error("Full traceback:")
            for line in traceback.format_exc().splitlines():
                self.logger.error(line)
            raise  # Re-raise memory errors as they are critical
    
    def _find_color_blobs(self, frame: np.ndarray, flip_blobs: bool = False, frame_idx: int = 0) -> List[np.ndarray]:
        """
        Find colored blobs in the frame with state tracking to prevent false detections.
        
        KEY BEHAVIOR:
        - When blob 1 has NOT disappeared: Always take the 2 LARGEST blobs by area (if more exist)
        - When blob 1 HAS disappeared: Only take the 1 LARGEST blob by area
        
        Args:
            frame: Input frame
            flip_blobs: Whether to flip blob ordering
            frame_idx: Current frame index for logging
            
        Returns:
            List of blob masks (max 2 before disappearance, max 1 after)
        """
        gray = frame.sum(axis=2)
        non_black = (gray > self.black_thresh)
        labeled = label(non_black, connectivity=2)
        regions = regionprops(labeled)
        
        # Sort by area and get all significant regions
        regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
        
        # Filter regions by minimum area to avoid noise
        min_area = 50  # Minimum blob area
        significant_regions = [r for r in regions_sorted if r.area > min_area]
        
        # Apply state-based blob detection logic
        if not self.blob_1_disappeared:
            # Normal case: can detect up to 2 blobs, always take the 2 largest by area
            if len(significant_regions) >= 2:
                # We have enough blobs - take the 2 largest by area
                top_regions = significant_regions[:2]
                self.blob_1_missing_count = 0  # Reset missing count
                if len(significant_regions) > 2:
                    self.logger.debug(f"Frame {frame_idx}: Found {len(significant_regions)} blobs, taking the 2 largest by area")
            else:
                # Not enough blobs detected (less than 2)
                self.blob_1_missing_count += 1
                self.logger.debug(f"Frame {frame_idx}: Only {len(significant_regions)} blobs detected, missing count: {self.blob_1_missing_count}")
                
                if self.blob_1_missing_count >= self.blob_1_missing_threshold:
                    # Mark blob 1 as disappeared
                    self.blob_1_disappeared = True
                    self.blob_1_disappeared_frame = frame_idx
                    self.logger.info(f"Frame {frame_idx}: Blob 1 marked as disappeared after {self.blob_1_missing_count} missing frames")
                
                # Use whatever regions we have (0 or 1 blob)
                top_regions = significant_regions[:2]  # This will be 0 or 1 blob
        else:
            # Blob 1 has disappeared - only allow detection of one blob (the largest)
            if len(significant_regions) >= 2:
                # Multiple blobs detected but blob 1 should be gone
                # Take only the largest blob by area
                top_regions = [significant_regions[0]]
                self.logger.debug(f"Frame {frame_idx}: Found {len(significant_regions)} blobs after blob 1 disappeared, taking only the largest by area")
            elif len(significant_regions) == 1:
                # One blob - normal case after disappearance
                top_regions = [significant_regions[0]]
            else:
                # No blobs detected
                top_regions = []
        
        # Sort by horizontal position if we have multiple regions
        if len(top_regions) > 1:
            reg_info = []
            for r in top_regions:
                coords = r.coords
                mean_col = coords[:, 1].mean()
                reg_info.append((r, mean_col))
            
            # Sort left-to-right or right-to-left based on flip_blobs
            reg_info.sort(key=lambda x: x[1], reverse=flip_blobs)
            
            # Extract masks
            masks = []
            for (r, _) in reg_info:
                mask = (labeled == r.label)
                masks.append(mask)
        else:
            # Single region or no regions
            masks = []
            for r in top_regions:
                mask = (labeled == r.label)
                masks.append(mask)
        
        return masks
    
    def _run_model_inference_with_splitting(self, frame: np.ndarray, H: int, W: int) -> List[np.ndarray]:
        """Run model inference and split connected components (following main_gen_vids_and_meshes.py)."""
        try:
            pil_image = Image.fromarray(frame, 'RGB')
            
            # Get predictions from model interface
            predictions = self.model_interface.infer_image(pil_image)
            pred_masks_tensor = predictions['pred_masks']  # (1, n_queries, H', W')
            
            # Convert to list of numpy masks and split connected components
            split_pred_masks = []
            for i in range(pred_masks_tensor.shape[1]):
                mask_tensor = pred_masks_tensor[0, i]  # (H', W')
                
                # Resize to original frame size if needed
                if mask_tensor.shape != (H, W):
                    mask_tensor = F.interpolate(
                        mask_tensor.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
                # Convert to binary mask
                binary_mask = mask_tensor.cpu().numpy() > 0.5
                
                # Split into connected components (IMPORTANT: This was missing proper implementation)
                labeled = label(binary_mask, connectivity=2)
                max_cc = labeled.max()
                for cc_label in range(1, max_cc + 1):
                    component_mask = (labeled == cc_label)
                    if component_mask.sum() > 0:  # Only add non-empty masks
                        split_pred_masks.append(component_mask)
            
            return split_pred_masks
            
        except Exception as e:
            import traceback
            self.logger.error(f"Failed model inference: {e}")
            self.logger.error("Full traceback:")
            for line in traceback.format_exc().splitlines():
                self.logger.error(line)
            # Return empty list on failure
            return []
    
    def _bipartite_assign_blobs_to_masks(self, blob_masks: List[np.ndarray], 
                                       pred_masks: List[np.ndarray]) -> Tuple[List[Optional[int]], Optional[np.ndarray]]:
        """Assign predicted masks to detected blobs using bipartite matching (from main_gen_vids_and_meshes.py)."""
        nb = len(blob_masks)
        np_ = len(pred_masks)
        
        if np_ == 0:
            return [None] * nb, None
        
        # Compute cost matrix (negative IoU)
        cost_matrix = np.zeros((nb, np_), dtype=np.float32)
        for b in range(nb):
            for p in range(np_):
                iou_val = self._compute_iou(blob_masks[b], pred_masks[p])
                cost_matrix[b, p] = -iou_val
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create assignment list
        assignments = [None] * nb
        for i in range(len(row_indices)):
            blob_idx = row_indices[i]
            pred_idx = col_indices[i]
            assignments[blob_idx] = pred_idx
        
        return assignments, cost_matrix
    
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union of two binary masks."""
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return 0.0 if union == 0 else intersection / union
    
    def _create_and_save_collage(self, frame: np.ndarray, blob_masks: List[np.ndarray], 
                               pred_masks: List[np.ndarray], cost_matrix: np.ndarray,
                               frame_idx: int, directories: Dict[str, str]) -> None:
        """Create and save collage showing top 10 mask assignments (from main_gen_vids_and_meshes.py)."""
        nb = len(blob_masks)
        
        # Create figure with subplots for each blob
        fig, axes = plt.subplots(nb, 10, figsize=(25, 5*nb), dpi=100)
        
        # Handle case where nb==1 (axes becomes 1D)
        if nb == 1 and len(axes.shape) == 1:
            axes = axes[np.newaxis, :]
        
        for b_idx in range(nb):
            row_cost = cost_matrix[b_idx, :]
            idx_sorted = np.argsort(row_cost)  # Sort by cost (negative IoU)
            best10 = idx_sorted[:10]  # Take top 10
            
            for rank_i, pred_idx in enumerate(best10):
                if rank_i >= 10:
                    break
                    
                ax = axes[b_idx, rank_i]
                overlay = frame.copy()
                
                # Green for the ground truth blob
                blob_mask_bool = blob_masks[b_idx].astype(bool)
                overlay[blob_mask_bool, 0] = 0
                overlay[blob_mask_bool, 1] = 255
                overlay[blob_mask_bool, 2] = 0
                
                # Red for the predicted mask
                if pred_idx < len(pred_masks):
                    pred_mask_bool = pred_masks[pred_idx].astype(bool)
                    overlay[pred_mask_bool, 0] = 255
                    overlay[pred_mask_bool, 1] = 0
                    overlay[pred_mask_bool, 2] = 0
                
                cost_val = row_cost[pred_idx]
                iou_val = -cost_val  # Convert back to positive IoU
                
                ax.imshow(overlay)
                ax.set_title(f"Blob {b_idx}, pred={pred_idx}\nIoU={iou_val:.3f}", fontsize=8)
                ax.set_axis_off()
        
        fig.suptitle(f"Frame {frame_idx} - Top 10 Mask Assignments", fontsize=14)
        collage_path = os.path.join(directories['frames_collage'], f"frame_{frame_idx:06d}_collage.png")
        fig.savefig(collage_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    def _create_memory_collage(self, frame: np.ndarray, assigned_masks: List[np.ndarray],
                             memory_masks: List[np.ndarray], frame_idx: int, 
                             directories: Dict[str, str]) -> None:
        """Create memory collage showing current vs memory masks (from main_gen_vids_and_meshes.py)."""
        try:
            nb = len(assigned_masks)
            
            fig, axes = plt.subplots(nb, 2, figsize=(10, 5*nb), dpi=100)
            if nb == 1 and len(axes.shape) == 1:
                axes = axes[np.newaxis, :]
            
            for b_i in range(nb):
                # Left: Current assigned mask
                axL = axes[b_i, 0]
                overlay_cur = frame.copy()
                if assigned_masks[b_i] is not None:
                    # Ensure mask is boolean for indexing
                    mask_bool = assigned_masks[b_i].astype(bool)
                    overlay_cur[mask_bool, 0] = 255
                    overlay_cur[mask_bool, 1] = 0
                    overlay_cur[mask_bool, 2] = 0
                axL.imshow(overlay_cur)
                axL.set_title(f"Blob {b_i} - Current", fontsize=8)
                axL.set_axis_off()

                # Right: Memory mask
                axR = axes[b_i, 1]
                overlay_mem = frame.copy()
                if b_i < len(memory_masks) and memory_masks[b_i] is not None:
                    # Ensure mask is boolean for indexing
                    mem_mask_bool = memory_masks[b_i].astype(bool)
                    overlay_mem[mem_mask_bool, 0] = 0
                    overlay_mem[mem_mask_bool, 1] = 255
                    overlay_mem[mem_mask_bool, 2] = 0
                axR.imshow(overlay_mem)
                axR.set_title(f"Blob {b_i} - Memory", fontsize=8)
                axR.set_axis_off()

            fig.suptitle(f"Frame {frame_idx} - Memory Collage", fontsize=14)
            memcoll_path = os.path.join(directories['frames_memory_collage'], f"frame_{frame_idx:06d}_memcollage.png")
            fig.savefig(memcoll_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to create memory collage for frame {frame_idx}: {e}")
            self.logger.error("Full traceback:")
            for line in traceback.format_exc().splitlines():
                self.logger.error(line)
            # Don't re-raise - continue processing
    
    def _update_memory_masks_with_strategy(self, assigned_masks: List[np.ndarray], frame_idx: int) -> None:
        """Update memory masks using different strategies for different blobs."""
        for i in range(self.n_blobs):
            if i < len(assigned_masks) and assigned_masks[i] is not None:
                new_mask = assigned_masks[i].astype(np.float32)
                
                # CRITICAL FIX: Check freeze condition FIRST for blob 1
                if i == 1 and self.blob_1_memory_freeze_frame is not None and frame_idx >= self.blob_1_memory_freeze_frame:
                    # Blob 1 memory is frozen - don't update AT ALL regardless of strategy
                    self.logger.debug(f"Frame {frame_idx}: Blob 1 memory frozen, not updating")
                    pass  # Keep existing memory
                elif i == 1 and self.blob_1_memory_strategy == 'running_average':
                    # Special handling for blob 1 with running average strategy (only if not frozen)
                    self._update_blob_1_running_average(new_mask, frame_idx)
                else:
                    # Standard exponential averaging for other blobs or blob 1 before freeze
                    self.mem_floats[i] = self.alpha * self.mem_floats[i] + (1 - self.alpha) * new_mask
    
    def _update_blob_1_running_average(self, new_mask: np.ndarray, frame_idx: int) -> None:
        """Update blob 1 memory using running average strategy."""
        # Note: Freeze frame check is now handled in the calling method
        
        # Add new mask to history
        self.blob_1_mask_history.append(new_mask.copy())
        
        # Compute running average
        if len(self.blob_1_mask_history) > 0:
            running_avg = np.mean(np.stack(list(self.blob_1_mask_history)), axis=0)
            self.mem_floats[1] = running_avg.astype(np.float32)
            
            self.logger.debug(f"Frame {frame_idx}: Updated blob 1 memory with running average of {len(self.blob_1_mask_history)} masks")
    
    def _get_memory_masks(self) -> List[np.ndarray]:
        """Get current memory masks as binary arrays."""
        masks = [mem_float > 0.5 for mem_float in self.mem_floats]
        
        # If blob 1 memory is frozen and we have a frozen mask, use that instead
        if (hasattr(self, 'current_frame_idx') and 
            self.blob_1_memory_freeze_frame is not None and 
            self.current_frame_idx >= self.blob_1_memory_freeze_frame and
            1 in self.frozen_memory_masks and
            len(masks) > 1):
            
            try:
                # Load the frozen mask
                frozen_mask_path = self.frozen_memory_masks[1]
                if os.path.exists(frozen_mask_path):
                    from PIL import Image
                    frozen_img = Image.open(frozen_mask_path).convert('L')
                    frozen_array = np.array(frozen_img, dtype=np.uint8)
                    frozen_mask = (frozen_array > 127).astype(np.float32)
                    masks[1] = frozen_mask
                    self.logger.debug(f"Using frozen mask for blob 1 at frame {self.current_frame_idx}")
            except Exception as e:
                self.logger.warning(f"Failed to load frozen mask for blob 1: {e}")
        
        return masks
    
    def _save_nonmemory_masks(self, assigned_masks: List[np.ndarray], frame_idx: int, 
                            directories: Dict[str, str]) -> None:
        """Save non-memory masks as PNG files."""
        for blob_idx, mask in enumerate(assigned_masks):
            if mask is not None and mask.sum() > 0:
                # Ensure mask is boolean before converting to uint8
                mask_bool = mask.astype(bool)
                mask_255 = (mask_bool.astype(np.uint8)) * 255
                mask_path = os.path.join(
                    directories['frames_masks_nonmem'],
                    f"mask_blob_{blob_idx}_frame_{frame_idx:06d}.png"
                )
                Image.fromarray(mask_255).save(mask_path)
    
    def _save_memory_masks_with_freeze_logic(self, memory_masks: List[np.ndarray], frame_idx: int,
                                           directories: Dict[str, str]) -> None:
        """Save memory masks with freeze frame logic - after freeze frame, reuse frozen masks."""
        
        for blob_idx, mask in enumerate(memory_masks):
            mask_path = os.path.join(
                directories['frames_masks'],
                f"mask_memory_blob_{blob_idx}_frame_{frame_idx:06d}.png"
            )
            
            # Check if this blob should be frozen
            if (blob_idx == 1 and 
                self.blob_1_memory_freeze_frame is not None and 
                frame_idx >= self.blob_1_memory_freeze_frame):
                
                # This is blob 1 and we're past the freeze frame
                if blob_idx in self.frozen_memory_masks:
                    # Copy the frozen mask file
                    frozen_mask_path = self.frozen_memory_masks[blob_idx]
                    if os.path.exists(frozen_mask_path):
                        shutil.copy2(frozen_mask_path, mask_path)
                        self.logger.debug(f"Frame {frame_idx}: Reused frozen blob 1 memory mask")
                    else:
                        self.logger.warning(f"Frame {frame_idx}: Frozen mask file not found: {frozen_mask_path}")
                        # Fall back to saving current mask but this should not happen
                        if mask.sum() > 0:
                            # Ensure mask is boolean before converting to uint8
                            mask_bool = mask.astype(bool)
                            mask_255 = (mask_bool.astype(np.uint8)) * 255
                            Image.fromarray(mask_255).save(mask_path)
                else:
                    # This shouldn't happen if logic is correct, but save current mask as fallback
                    self.logger.warning(f"Frame {frame_idx}: No frozen mask stored for blob 1")
                    if mask.sum() > 0:
                        # Ensure mask is boolean before converting to uint8
                        mask_bool = mask.astype(bool)
                        mask_255 = (mask_bool.astype(np.uint8)) * 255
                        Image.fromarray(mask_255).save(mask_path)
            
            elif (blob_idx == 1 and 
                  self.blob_1_memory_freeze_frame is not None and 
                  frame_idx == self.blob_1_memory_freeze_frame - 1):
                
                # This is the last frame before freezing - save as the frozen mask
                if mask.sum() > 0:
                    # Ensure mask is boolean before converting to uint8
                    mask_bool = mask.astype(bool)
                    mask_255 = (mask_bool.astype(np.uint8)) * 255
                    Image.fromarray(mask_255).save(mask_path)
                    
                    # Store this as the frozen mask to reuse
                    self.frozen_memory_masks[blob_idx] = mask_path
                    self.logger.info(f"Frame {frame_idx}: Saved blob 1 memory mask to be frozen at {mask_path}")
            
            else:
                # Normal case - save the current mask
                if mask.sum() > 0:
                    # Ensure mask is boolean before converting to uint8
                    mask_bool = mask.astype(bool)
                    mask_255 = (mask_bool.astype(np.uint8)) * 255
                    Image.fromarray(mask_255).save(mask_path)
    
    def _save_blob_visualization(self, frame: np.ndarray, blob_masks: List[np.ndarray],
                               frame_idx: int, directories: Dict[str, str]) -> None:
        """Save visualization of detected blobs."""
        debug_frame = frame.astype(np.float32).copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        for i, blob_mask in enumerate(blob_masks):
            color = colors[i % len(colors)]
            # Ensure blob_mask is boolean for indexing
            blob_mask_bool = blob_mask.astype(bool)
            debug_frame[blob_mask_bool, 0] = color[0]
            debug_frame[blob_mask_bool, 1] = color[1] 
            debug_frame[blob_mask_bool, 2] = color[2]
        
        debug_path = os.path.join(directories['frames_blobs'], f"frame_{frame_idx:06d}_blobs.png")
        Image.fromarray(debug_frame.astype(np.uint8)).save(debug_path)
    
    def _create_and_save_final_overlay(self, frame: np.ndarray, memory_masks: List[np.ndarray],
                                     frame_idx: int, directories: Dict[str, str], flip_blobs: bool,
                                     H: int, W: int) -> None:
        """Create final overlay with polygons and save processed frame."""
        
        # Make masks disjoint (from main_gen_vids_and_meshes.py)
        disjoint_masks = self._make_masks_disjoint(memory_masks.copy())
        
        # Create overlay
        overlay_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(overlay_img, "RGBA")
        
        # Colors for different blobs
        color_list = [
            (255, 0, 0, 100),    # Red
            (0, 255, 0, 100),    # Green  
            (0, 0, 255, 100),    # Blue
            (255, 255, 0, 100),  # Yellow
            (255, 0, 255, 100),  # Magenta
        ]
        text_fill = (255, 255, 255, 255)  # White text
        
        cx_ = W / 2.0
        cy_ = H / 2.0
        
        # Sort masks by position for consistent ordering
        mask_info = []
        for i, mask in enumerate(disjoint_masks):
            if mask is not None and mask.sum() > 0:
                # Ensure mask is boolean for argwhere
                mask_bool = mask.astype(bool)
                coords = np.argwhere(mask_bool)
                mean_col = coords[:, 1].mean()
                mask_info.append((i, mask, mean_col))
            else:
                mask_info.append((i, None, 999999))
        
        # Sort by position (flip if needed)
        mask_info.sort(key=lambda x: x[2], reverse=flip_blobs)
        
        # Draw memory masks as colored overlays with polygons
        for order_i, (orig_i, mask, _) in enumerate(mask_info):
            if mask is None or mask.sum() == 0:
                continue
                
            # Create polygon from mask contours
            # Ensure mask is boolean before converting to uint8
            mask_bool = mask.astype(bool)
            contours = find_contours(mask_bool.astype(np.uint8), 0.5)
            if contours:
                largest_contour = max(contours, key=len)
                polygon_points = []
                for point in largest_contour:
                    r = point[0]
                    c = point[1]
                    x = c - cx_
                    y = r - cy_
                    polygon_points.append((x + cx_, y + cy_))
                
                if len(polygon_points) > 2:
                    draw.polygon(polygon_points, fill=color_list[order_i % len(color_list)])
                    
                    # Add label at centroid
                    centroid_x = sum(p[0] for p in polygon_points) / len(polygon_points)
                    centroid_y = sum(p[1] for p in polygon_points) / len(polygon_points)
                    draw.text((centroid_x, centroid_y), f"Blob {order_i}", fill=text_fill)
        
        # Save processed frame
        output_path = os.path.join(directories['frames_processed'], f"frame_{frame_idx:06d}.png")
        overlay_img.save(output_path)
    
    def _make_masks_disjoint(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """Make masks disjoint by removing overlaps (from main_gen_vids_and_meshes.py)."""
        for i in range(len(masks)):
            if masks[i] is None:
                continue
            # Ensure mask_i is boolean for bitwise operations
            mask_i_bool = masks[i].astype(bool)
            for j in range(i+1, len(masks)):
                if masks[j] is None:
                    continue
                # Ensure mask_j is boolean for bitwise operations
                mask_j_bool = masks[j].astype(bool)
                masks[j] = mask_j_bool & ~mask_i_bool
        return masks
    
    def _create_final_video(self, directories: Dict[str, str], video_metadata: Dict[str, Any]) -> None:
        """Create final video from processed frames."""
        video_name = os.path.basename(directories['root']) + ".mp4"
        final_video_path = os.path.join(directories['videos_processed'], video_name)
        
        self.logger.info(f"Creating final video: {final_video_path}")
        
        # Find all available processed frames
        frames_processed_dir = directories['frames_processed']
        available_frames = []
        for frame_idx in range(video_metadata['total_frames']):
            frame_path = os.path.join(frames_processed_dir, f"frame_{frame_idx:06d}.png")
            if os.path.exists(frame_path):
                available_frames.append((frame_idx, frame_path))
        
        if not available_frames:
            self.logger.error("No processed frames found for video creation")
            return
        
        self.logger.info(f"Found {len(available_frames)} processed frames out of {video_metadata['total_frames']} total frames")
        
        writer = imageio.get_writer(final_video_path, fps=video_metadata['fps'], macro_block_size=1)
        
        try:
            frames_written = 0
            for frame_idx, frame_path in available_frames:
                frame = imageio.v2.imread(frame_path)
                writer.append_data(frame)
                frames_written += 1
                
                if frames_written % 100 == 0:
                    self.logger.info(f"Written {frames_written}/{len(available_frames)} frames to video")
        finally:
            writer.close()
        
        self.logger.info(f"Final video created successfully with {frames_written} frames")
    
    def _count_existing_frames(self, frames_processed_dir: str) -> int:
        """Count existing processed frames."""
        if not os.path.exists(frames_processed_dir):
            return 0
        
        frame_files = [f for f in os.listdir(frames_processed_dir) 
                      if f.startswith("frame_") and f.endswith(".png")]
        return len(frame_files)
    
    def _attempt_video_generation_from_existing_frames(self, directories: Dict[str, str], video_metadata: Dict[str, Any]) -> bool:
        """
        Attempt to generate video from existing frames.
        Returns True if successful, False if not enough frames.
        """
        frames_count = self._count_existing_frames(directories['frames_processed'])
        min_required_frames = max(10, video_metadata['total_frames'] // 2)  # Need at least half the frames or 10 frames
        
        if frames_count >= min_required_frames:
            self.logger.info(f"Found {frames_count} existing frames, attempting video generation...")
            try:
                self._create_final_video(directories, video_metadata)
                return True
            except Exception as e:
                self.logger.error(f"Failed to create video from existing frames: {e}")
                return False
        else:
            self.logger.info(f"Not enough frames for video generation: {frames_count} < {min_required_frames}")
            return False
    
    # Helper methods
    def _parse_video_prefix(self, video_path: str) -> str:
        """Extract video prefix from path."""
        base = os.path.basename(video_path)
        root, _ = os.path.splitext(base)
        return root.replace(" ", "+")
    
    def _video_is_flipped(self, video_path: str) -> bool:
        """Check if video is flipped based on filename."""
        base = os.path.basename(video_path)
        root, _ = os.path.splitext(base)
        return "flipped" in root
    
    def _normalize_frame_dimensions(self, frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Ensure frame has correct dimensions."""
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            corrected = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
            h_min = min(target_h, frame.shape[0])
            w_min = min(target_w, frame.shape[1])
            corrected[:h_min, :w_min, :] = frame[:h_min, :w_min, :]
            return corrected
        return frame
    

    
    def _count_processed_frames(self, frames_dir: str) -> int:
        """Count number of processed frames in directory."""
        try:
            frame_files = [f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")]
            return len(frame_files)
        except (OSError, FileNotFoundError):
            return 0