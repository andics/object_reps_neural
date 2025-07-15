#!/usr/bin/env python3
"""
vanilla_segmentation.py

Modular vanilla segmentation component that saves raw model segmentation results
without any blob matching, memory, or post-processing. Can be used across all experiments.

Usage:
    from vanilla_segmentation import VanillaSegmentationSaver
    
    saver = VanillaSegmentationSaver(model_interface, output_dir)
    saver.save_frame_segmentation(frame, frame_idx)
"""

import os
import logging
from pathlib import Path
from typing import Union, Optional
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from segformer.segformer_interface import ModelInterface


class VanillaSegmentationSaver:
    """
    Saves vanilla segmentation results for frames without any post-processing.
    
    This component:
    1. Takes raw frames and runs model inference
    2. Saves raw segmentation maps and masks
    3. Creates visualization overlays
    4. Works with any ModelInterface implementation
    """
    
    def __init__(self, model_interface: ModelInterface, output_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize vanilla segmentation saver.
        
        Args:
            model_interface: The model interface to use for inference
            output_dir: Output directory for saving segmentation results
            logger: Optional logger instance
        """
        self.model_interface = model_interface
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        
        # Create output directory structure
        self.seg_maps_dir = self.output_dir / "segmentation_maps"
        self.raw_masks_dir = self.output_dir / "raw_masks"
        self.overlays_dir = self.output_dir / "overlays"
        self.visualizations_dir = self.output_dir / "visualizations"
        
        # Create all directories
        for dir_path in [self.seg_maps_dir, self.raw_masks_dir, self.overlays_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.logger.info(f"VanillaSegmentationSaver initialized with output dir: {self.output_dir}")
    
    def save_frame_segmentation(self, frame: Union[np.ndarray, Image.Image], frame_idx: int) -> dict:
        """
        Process a single frame and save vanilla segmentation results.
        
        Args:
            frame: Input frame as numpy array (H, W, 3) or PIL Image
            frame_idx: Frame index for file naming
            
        Returns:
            Dictionary with paths to saved files and segmentation info
        """
        # Convert frame to PIL Image if numpy array
        if isinstance(frame, np.ndarray):
            pil_image = Image.fromarray(frame).convert('RGB')
        else:
            pil_image = frame.convert('RGB')
        
        orig_width, orig_height = pil_image.size
        
        # Load model if not already loaded
        if not hasattr(self.model_interface, 'model') or self.model_interface.model is None:
            self.logger.info("Loading model for vanilla segmentation...")
            self.model_interface.load_model()
        
        # Run model inference
        try:
            predictions = self.model_interface.infer_image(pil_image)
        except Exception as e:
            self.logger.error(f"Model inference failed for frame {frame_idx}: {e}")
            return {}
        
        # Get raw segmentation map if available (for SegFormer)
        seg_map = None
        if hasattr(self.model_interface, 'processor') and hasattr(self.model_interface, 'model'):
            try:
                # Run raw inference to get segmentation map
                pixel_values = self.model_interface.processor(pil_image, return_tensors="pt").pixel_values.to(self.model_interface.device)
                with torch.no_grad():
                    outputs = self.model_interface.model(pixel_values)
                
                seg_map = self.model_interface.processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(orig_height, orig_width)]
                )[0]  # Shape: (H, W)
                
            except Exception as e:
                self.logger.warning(f"Could not extract segmentation map: {e}")
        
        # Save results
        result_paths = {}
        
        # 1. Save raw masks from predictions
        pred_masks = predictions.get('pred_masks')  # (1, num_queries, H, W)
        if pred_masks is not None:
            # Save individual masks
            for i in range(pred_masks.shape[1]):
                mask = pred_masks[0, i].cpu().numpy()  # (H, W)
                if mask.sum() > 0:  # Only save non-empty masks
                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
                    mask_path = self.raw_masks_dir / f"frame_{frame_idx:06d}_mask_{i:03d}.png"
                    Image.fromarray(mask_binary).save(mask_path)
                    
                    if i == 0:  # Store path for first mask as example
                        result_paths['first_mask'] = str(mask_path)
        
        # 2. Save segmentation map if available
        if seg_map is not None:
            # Save raw segmentation map
            seg_map_np = seg_map.cpu().numpy().astype(np.uint8)
            seg_map_path = self.seg_maps_dir / f"frame_{frame_idx:06d}_segmap.png"
            Image.fromarray(seg_map_np).save(seg_map_path)
            result_paths['segmentation_map'] = str(seg_map_path)
            
            # Create colored overlay
            if hasattr(self.model_interface, 'ade_palette'):
                palette = np.array(self.model_interface.ade_palette(), dtype=np.uint8)
                seg_colored = palette[seg_map_np]  # (H, W, 3)
                
                # Create blended overlay
                frame_array = np.array(pil_image)
                blend = (0.6 * frame_array + 0.4 * seg_colored[..., ::-1]).astype(np.uint8)
                
                overlay_path = self.overlays_dir / f"frame_{frame_idx:06d}_overlay.png"
                Image.fromarray(blend).save(overlay_path)
                result_paths['overlay'] = str(overlay_path)
        
        # 3. Create comprehensive visualization
        self._create_visualization(pil_image, predictions, seg_map, frame_idx, result_paths)
        
        self.logger.debug(f"Saved vanilla segmentation for frame {frame_idx}")
        return result_paths
    
    def _create_visualization(self, image: Image.Image, predictions: dict, 
                            seg_map: Optional[torch.Tensor], frame_idx: int, result_paths: dict) -> None:
        """Create a comprehensive visualization of the segmentation results."""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            # Plot original image
            axes[0].imshow(image)
            axes[0].set_title(f'Original Frame {frame_idx}')
            axes[0].axis('off')
            
            # Plot segmentation map if available
            if seg_map is not None:
                axes[1].imshow(seg_map.cpu().numpy(), cmap='tab20')
                axes[1].set_title('Segmentation Map')
                axes[1].axis('off')
            else:
                axes[1].text(0.5, 0.5, 'No segmentation map', ha='center', va='center', transform=axes[1].transAxes)
                axes[1].axis('off')
            
            # Plot combined masks if available
            pred_masks = predictions.get('pred_masks')
            if pred_masks is not None:
                combined_mask = torch.zeros_like(pred_masks[0, 0])
                mask_count = 0
                for i in range(min(10, pred_masks.shape[1])):
                    mask = pred_masks[0, i]
                    if mask.sum() > 0:
                        combined_mask += mask * (mask_count + 1)
                        mask_count += 1
                        if mask_count >= 5:  # Limit to 5 masks for visibility
                            break
                
                axes[2].imshow(combined_mask.cpu().numpy(), cmap='tab10')
                axes[2].set_title(f'Combined Masks (Top {mask_count})')
                axes[2].axis('off')
            else:
                axes[2].text(0.5, 0.5, 'No masks available', ha='center', va='center', transform=axes[2].transAxes)
                axes[2].axis('off')
            
            # Plot overlay if available
            if 'overlay' in result_paths:
                overlay_img = Image.open(result_paths['overlay'])
                axes[3].imshow(overlay_img)
                axes[3].set_title('Segmentation Overlay')
                axes[3].axis('off')
            else:
                axes[3].text(0.5, 0.5, 'No overlay available', ha='center', va='center', transform=axes[3].transAxes)
                axes[3].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.visualizations_dir / f"frame_{frame_idx:06d}_visualization.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            result_paths['visualization'] = str(viz_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to create visualization for frame {frame_idx}: {e}")
    
    def process_video_frames(self, video_path: str, start_frame: int = 0, max_frames: Optional[int] = None) -> dict:
        """
        Process all frames in a video and save vanilla segmentation results.
        
        Args:
            video_path: Path to input video file
            start_frame: Starting frame index
            max_frames: Maximum number of frames to process (None for all)
            
        Returns:
            Dictionary with processing statistics
        """
        import imageio
        
        self.logger.info(f"Processing video frames from: {video_path}")
        
        try:
            reader = imageio.get_reader(video_path, format='ffmpeg')
            
            frame_count = 0
            processed_count = 0
            
            try:
                while True:
                    if max_frames and processed_count >= max_frames:
                        break
                        
                    try:
                        frame = reader.get_data(start_frame + frame_count)
                    except IndexError:
                        break
                    
                    # Process frame
                    result = self.save_frame_segmentation(frame, start_frame + frame_count)
                    if result:
                        processed_count += 1
                    
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        self.logger.info(f"Processed {processed_count} frames")
                        
            finally:
                reader.close()
                
            stats = {
                'total_frames_read': frame_count,
                'frames_processed': processed_count,
                'output_directory': str(self.output_dir)
            }
            
            self.logger.info(f"Video processing complete: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to process video: {e}")
            return {}
