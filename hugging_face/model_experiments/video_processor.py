#!/usr/bin/env python3
"""
video_processor.py

Comprehensive video processing class that handles video frame extraction,
model inference, mask generation, and output creation. 
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

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


class VideoProcessor:
    """
    Comprehensive video processor that handles:
    - Video frame extraction and processing
    - Model inference using configurable interfaces
    - Blob detection and mask assignment
    - Memory-based mask tracking
    - Output generation (masks, visualizations, videos)
    """
    
    def __init__(self, model_interface: ModelInterface, logger: logging.Logger = None):
        self.model_interface = model_interface
        self.logger = logger or logging.getLogger(__name__)
        
        # Processing parameters
        self.n_blobs = 2
        self.initial_skip_frames = 13
        self.alpha = 0.7  # Memory decay factor
        self.black_thresh = 30  # Threshold for blob detection
        
        # Memory for tracking masks across frames
        self.mem_floats = None
        self.current_video_shape = None
        
    def setup_output_directories(self, video_path: str, output_root: str, model_prefix: str = None) -> Dict[str, str]:
        """Setup organized output directory structure for video processing."""
        
        if model_prefix is None:
            model_prefix = "segformer_model"
            
        video_prefix = self._parse_video_prefix(video_path)
        
        # Create root folder structure
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
            'metadata': os.path.join(root_folder, "metadata")
        }
        
        # Create all directories
        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)
            
        return directories
    
    def check_processing_status(self, directories: Dict[str, str], total_frames: int = None) -> Dict[str, Any]:
        """Check which processing steps have been completed."""
        status = {
            'frames_extracted': False,
            'masks_generated': False,
            'video_created': False,
            'last_processed_frame': -1,
            'total_frames': total_frames,
            'can_resume': False
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
        
        # Can resume if we have some processed frames but not complete
        if total_frames:
            status['can_resume'] = 0 < processed_frames < total_frames
        
        return status
    
    def process_video(self, video_path: str, output_root: str, model_prefix: str = None, 
                     resume: bool = True) -> Dict[str, str]:
        """Process a complete video through the full pipeline."""
        self.logger.info(f"Starting video processing: {video_path}")
        
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
        
        if resume and status['can_resume']:
            self.logger.info(f"Resuming processing from frame {status['last_processed_frame'] + 1}")
            start_frame = status['last_processed_frame'] + 1
        else:
            self.logger.info("Starting processing from beginning")
            start_frame = 0
            # Initialize memory
            self._initialize_memory(video_metadata['height'], video_metadata['width'])
        
        # Process frames
        self._process_video_frames(
            video_path, directories, video_metadata, start_frame
        )
        
        # Create final video if not exists
        if not status['video_created']:
            self._create_final_video(directories, video_metadata)
        
        self.logger.info(f"Video processing completed: {directories['root']}")
        return directories
    
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
        
        try:
            for frame_idx in range(start_frame, video_metadata['total_frames']):
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
                
                # Log progress
                if frame_idx % 10 == 0:
                    self.logger.info(f"Processed frame {frame_idx}/{video_metadata['total_frames']}")
        
        finally:
            reader.close()
    
    def _process_single_frame(self, frame: np.ndarray, frame_idx: int, 
                            directories: Dict[str, str], flip_blobs: bool, H: int, W: int) -> None:
        """Process a single video frame."""
        
        # Skip detection for initial frames
        if frame_idx < self.initial_skip_frames:
            self._save_skipped_frame(frame, frame_idx, directories)
            return
        
        # 1. Detect color blobs
        blob_masks = self._find_color_blobs(frame, flip_blobs)
        
        if len(blob_masks) == 0:
            self._save_skipped_frame(frame, frame_idx, directories)
            return
        
        # Save blob visualization
        self._save_blob_visualization(frame, blob_masks, frame_idx, directories)
        
        # 2. Run model inference
        pred_masks = self._run_model_inference(frame, H, W)
        
        # 3. Assign masks to blobs
        assigned_masks, cost_matrix = self._assign_masks_to_blobs(blob_masks, pred_masks)
        
        # 4. Save non-memory masks
        self._save_nonmemory_masks(assigned_masks, frame_idx, directories)
        
        # 5. Update memory
        self._update_memory_masks(assigned_masks)
        
        # 6. Get memory masks
        memory_masks = self._get_memory_masks()
        
        # 7. Save memory masks
        self._save_memory_masks(memory_masks, frame_idx, directories)
        
        # 8. Create final overlay and save
        self._create_and_save_overlay(frame, memory_masks, frame_idx, directories, flip_blobs, H, W)
    
    def _find_color_blobs(self, frame: np.ndarray, flip_blobs: bool = False) -> List[np.ndarray]:
        """Find colored blobs in the frame."""
        gray = frame.sum(axis=2)
        non_black = (gray > self.black_thresh)
        labeled = label(non_black, connectivity=2)
        regions = regionprops(labeled)
        
        # Sort by area and take top n_blobs
        regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
        top_regions = regions_sorted[:self.n_blobs]
        
        # Sort by horizontal position
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
        
        return masks
    
    def _run_model_inference(self, frame: np.ndarray, H: int, W: int) -> List[np.ndarray]:
        """Run model inference on the frame."""
        pil_image = Image.fromarray(frame, 'RGB')
        
        # Get predictions from model interface
        predictions = self.model_interface.infer_image(pil_image)
        pred_masks_tensor = predictions['pred_masks']  # (1, n_queries, H', W')
        
        # Convert to list of numpy masks
        pred_masks = []
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
            
            # Split into connected components
            labeled = label(binary_mask, connectivity=2)
            for cc_label in range(1, labeled.max() + 1):
                component_mask = (labeled == cc_label)
                pred_masks.append(component_mask)
        
        return pred_masks
    
    def _assign_masks_to_blobs(self, blob_masks: List[np.ndarray], 
                             pred_masks: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        """Assign predicted masks to detected blobs using bipartite matching."""
        n_blobs = len(blob_masks)
        n_preds = len(pred_masks)
        
        if n_preds == 0:
            return [None] * n_blobs, None
        
        # Compute cost matrix (negative IoU)
        cost_matrix = np.zeros((n_blobs, n_preds), dtype=np.float32)
        for b in range(n_blobs):
            for p in range(n_preds):
                iou_val = self._compute_iou(blob_masks[b], pred_masks[p])
                cost_matrix[b, p] = -iou_val
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Create assignment list
        assignments = [None] * n_blobs
        for i in range(len(row_indices)):
            blob_idx = row_indices[i]
            pred_idx = col_indices[i]
            assignments[blob_idx] = pred_masks[pred_idx]
        
        return assignments, cost_matrix
    
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union of two binary masks."""
        intersection = (mask1 & mask2).sum()
        union = (mask1 | mask2).sum()
        return 0.0 if union == 0 else intersection / union
    
    def _update_memory_masks(self, assigned_masks: List[np.ndarray]) -> None:
        """Update memory masks with exponential decay."""
        for i in range(self.n_blobs):
            if i < len(assigned_masks) and assigned_masks[i] is not None:
                new_mask = assigned_masks[i].astype(np.float32)
                self.mem_floats[i] = self.alpha * self.mem_floats[i] + (1 - self.alpha) * new_mask
    
    def _get_memory_masks(self) -> List[np.ndarray]:
        """Get current memory masks as binary arrays."""
        return [mem_float > 0.5 for mem_float in self.mem_floats]
    
    def _save_nonmemory_masks(self, assigned_masks: List[np.ndarray], frame_idx: int, 
                            directories: Dict[str, str]) -> None:
        """Save non-memory masks as PNG files."""
        for blob_idx, mask in enumerate(assigned_masks):
            if mask is not None and mask.sum() > 0:
                mask_255 = (mask.astype(np.uint8)) * 255
                mask_path = os.path.join(
                    directories['frames_masks_nonmem'],
                    f"mask_blob_{blob_idx}_frame_{frame_idx:06d}.png"
                )
                Image.fromarray(mask_255).save(mask_path)
    
    def _save_memory_masks(self, memory_masks: List[np.ndarray], frame_idx: int,
                         directories: Dict[str, str]) -> None:
        """Save memory masks as PNG files."""
        for blob_idx, mask in enumerate(memory_masks):
            if mask.sum() > 0:
                mask_255 = (mask.astype(np.uint8)) * 255
                mask_path = os.path.join(
                    directories['frames_masks'],
                    f"mask_memory_blob_{blob_idx}_frame_{frame_idx:06d}.png"
                )
                Image.fromarray(mask_255).save(mask_path)
    
    def _save_blob_visualization(self, frame: np.ndarray, blob_masks: List[np.ndarray],
                               frame_idx: int, directories: Dict[str, str]) -> None:
        """Save visualization of detected blobs."""
        debug_frame = frame.astype(np.float32).copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        for i, blob_mask in enumerate(blob_masks):
            color = colors[i % len(colors)]
            debug_frame[blob_mask, 0] = color[0]
            debug_frame[blob_mask, 1] = color[1] 
            debug_frame[blob_mask, 2] = color[2]
        
        debug_path = os.path.join(directories['frames_blobs'], f"frame_{frame_idx:06d}_blobs.png")
        Image.fromarray(debug_frame.astype(np.uint8)).save(debug_path)
    
    def _create_and_save_overlay(self, frame: np.ndarray, memory_masks: List[np.ndarray],
                               frame_idx: int, directories: Dict[str, str], flip_blobs: bool,
                               H: int, W: int) -> None:
        """Create final overlay with polygons and save processed frame."""
        
        # Create overlay
        overlay_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(overlay_img, "RGBA")
        
        # Colors for different blobs
        colors = [
            (255, 0, 0, 100),    # Red
            (0, 255, 0, 100),    # Green  
            (0, 0, 255, 100),    # Blue
            (255, 255, 0, 100),  # Yellow
        ]
        
        # Draw memory masks as colored overlays
        for i, mask in enumerate(memory_masks):
            if mask.sum() > 0:
                # Create polygon from mask contours
                contours = find_contours(mask.astype(np.uint8), 0.5)
                if contours:
                    largest_contour = max(contours, key=len)
                    polygon_points = [(int(point[1]), int(point[0])) for point in largest_contour]
                    
                    if len(polygon_points) > 2:
                        draw.polygon(polygon_points, fill=colors[i % len(colors)])
                        
                        # Add label at centroid
                        centroid_x = sum(p[0] for p in polygon_points) / len(polygon_points)
                        centroid_y = sum(p[1] for p in polygon_points) / len(polygon_points)
                        draw.text((centroid_x, centroid_y), f"Blob {i}", fill=(255, 255, 255, 255))
        
        # Save processed frame
        output_path = os.path.join(directories['frames_processed'], f"frame_{frame_idx:06d}.png")
        overlay_img.save(output_path)
    
    def _create_final_video(self, directories: Dict[str, str], video_metadata: Dict[str, Any]) -> None:
        """Create final video from processed frames."""
        video_name = os.path.basename(directories['root']) + ".mp4"
        final_video_path = os.path.join(directories['videos_processed'], video_name)
        
        self.logger.info(f"Creating final video: {final_video_path}")
        
        writer = imageio.get_writer(final_video_path, fps=video_metadata['fps'], macro_block_size=1)
        
        try:
            for frame_idx in range(video_metadata['total_frames']):
                frame_path = os.path.join(directories['frames_processed'], f"frame_{frame_idx:06d}.png")
                if os.path.exists(frame_path):
                    frame = imageio.v2.imread(frame_path)
                    writer.append_data(frame)
        finally:
            writer.close()
        
        self.logger.info("Final video created successfully")
    
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
    
    def _save_skipped_frame(self, frame: np.ndarray, frame_idx: int, directories: Dict[str, str]) -> None:
        """Save frame without processing (for skipped frames)."""
        output_path = os.path.join(directories['frames_processed'], f"frame_{frame_idx:06d}.png")
        Image.fromarray(frame).save(output_path)
    
    def _count_processed_frames(self, frames_dir: str) -> int:
        """Count number of processed frames in directory."""
        try:
            frame_files = [f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")]
            return len(frame_files)
        except (OSError, FileNotFoundError):
            return 0