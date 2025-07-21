#!/usr/bin/env python3
"""
exp2TTC.py

Time-to-Collision (TTC) Experiment that processes raw .mp4 videos and computes collision detection
times under varying IoU thresholds, correlating with participant response data.
Completely self-contained from raw videos to final analysis.

FIXED VERSION:
- Enhanced blob 1 memory freezing logic for true stability
- Fixed undefined variable error in data analysis
- Added robust correlation computation with NaN protection
- Improved concave vs convex analysis with two-column bar charts using gen_two_box_plots.py style

Usage:
    python exp2TTC.py --model_interface segformer --videos_dir /path/to/raw_videos --csv_path /path/to/participants.csv --output_dir /path/to/output [--resume] [--blob_1_memory_freeze_frame 80]
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
from PIL import Image
import glob

# Import model interfaces and video processor
from segformer.segformer_interface import SegFormerInterface, ModelInterface
from video_processor import VideoProcessor

##############################################################################
# EXPERIMENT CLASS
##############################################################################

class TTCExperiment:
    """
    Experiment 2: Time-to-Collision Analysis
    
    This experiment:
    1. Takes a directory of raw .mp4 video files as input
    2. Processes each video using VideoProcessor to extract frames and detect objects
    3. Computes collision times under varying IoU thresholds
    4. Correlates model predictions with participant response times
    5. Generates analysis plots and statistics
    
    Special configuration for TTC experiment:
    - Blob 1 memory uses running average of past 10 frames
    - Blob 1 memory updates stop after specified freeze frame (default: 80)
    """

    def __init__(self, model_interface: ModelInterface, output_dir: str, n_blobs: int = 2,
                 blob_1_memory_freeze_frame: int = 80, logger: logging.Logger = None):
        self.model_interface = model_interface
        self.output_dir = output_dir
        self.blob_1_memory_freeze_frame = blob_1_memory_freeze_frame

        # Create output subdirectories FIRST (before logger is set up)
        self.results_dir = os.path.join(output_dir, "results")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.processed_videos_dir = os.path.join(output_dir, "processed_videos")

        for dir_path in [self.results_dir, self.plots_dir, self.logs_dir, self.processed_videos_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Now setup logger after logs_dir exists
        self.logger = logger or self._setup_logger()

        # Initialize video processor with TTC-specific memory strategy
        # - Blob 1 uses running average of past 10 frames
        # - Blob 1 memory updates stop after specified freeze frame
        self.video_processor = VideoProcessor(
            model_interface=model_interface, 
            n_blobs=n_blobs, 
            logger=self.logger,
            blob_1_memory_strategy='running_average',
            blob_1_running_avg_window=10,
            blob_1_memory_freeze_frame=self.blob_1_memory_freeze_frame
        )

        self.logger.info(f"Initialized TTC Experiment with output dir: {output_dir}")
        self.logger.info("TTC-specific configuration:")
        self.logger.info("  - Blob 1 memory: running average of past 10 frames")
        self.logger.info(f"  - Blob 1 memory updates stop after frame {self.blob_1_memory_freeze_frame}")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(self.logs_dir, f"ttc_exp_{timestamp}.log")

        logger = logging.getLogger(f"ttc_exp_{timestamp}")
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

    def run_full_experiment(self, videos_dir: str, csv_path: str, 
                          iou_start: float = 0.05, iou_end: float = 0.95, iou_step: float = 0.05,
                          resume: bool = True) -> None:
        """Run the complete TTC experiment from raw videos to final analysis."""
        self.logger.info("Starting full TTC experiment")
        
        # Step 1: Find all .mp4 video files in the input directory
        video_files = self._find_video_files(videos_dir)
        if not video_files:
            self.logger.error(f"No .mp4 video files found in {videos_dir}")
            return
        
        self.logger.info(f"Found {len(video_files)} video files to process")
        
        # Step 2: Load participant data
        participant_df = self._read_participant_csv(csv_path)
        
        # Step 3: Generate collision detection data  
        iou_values = np.arange(iou_start, iou_end + iou_step, iou_step)
        iou_values = np.round(iou_values, decimals=3)
        
        collision_data = self._process_videos_for_collision_detection(video_files, iou_values, resume=resume)
        
        # Step 4: ALWAYS analyze correlations with participant data (collect from all sources)
        # Collect collision data from all videos (both newly processed and existing)
        all_video_names = [Path(video_file).stem for video_file in video_files]
        self.logger.info("Collecting collision data from all videos for analysis...")
        final_collision_data = self._collect_all_collision_data(all_video_names, iou_values)
        
        if final_collision_data:
            self.logger.info(f"Collected collision data for {len(set([vname for (vname, _) in final_collision_data.keys()]))} videos")
            self._analyze_participant_correlations(final_collision_data, participant_df, iou_values)
        else:
            self.logger.warning("No collision data available from any videos - skipping correlation analysis")
        
        self.logger.info("TTC experiment completed successfully")

    def _find_video_files(self, videos_dir: str) -> List[str]:
        """Find all .mp4 video files in the specified directory."""
        video_pattern = os.path.join(videos_dir, "*.mp4")
        video_files = glob.glob(video_pattern)
        return sorted(video_files)

    def _read_participant_csv(self, csv_path: str) -> pd.DataFrame:
        """Read the participant CSV."""
        self.logger.info(f"Reading participant CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
        self.logger.info(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns.")
        return df

    def _process_videos_for_collision_detection(self, video_files: List[str], 
                                              iou_values: np.ndarray, resume: bool = True) -> Dict[Tuple[str, float], float]:
        """Process all videos to detect collision times for different IoU thresholds."""
        self.logger.info("Starting collision detection across videos and IoU thresholds...")
        
        collision_times = {}
        
        # First, collect ALL videos that should have collision data (both processed and to be processed)
        all_video_names = [Path(video_file).stem for video_file in video_files]
        
        for video_file in video_files:
            video_name = Path(video_file).stem
            self.logger.info(f"Processing video: {video_name}")
            
            # Check processing status
            expected_output_dir = os.path.join(self.processed_videos_dir, f"segformer_model-{video_name}")
            videos_processed_dir = os.path.join(expected_output_dir, "videos_processed")
            frames_processed_dir = os.path.join(expected_output_dir, "frames_processed")
            
            # Check if final video already exists
            if resume and os.path.exists(videos_processed_dir):
                video_files_in_dir = [f for f in os.listdir(videos_processed_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
                if video_files_in_dir:
                    self.logger.info(f"Video {video_name} already processed (found {video_files_in_dir[0]}), skipping video processing...")
                    # Don't skip completely - we still need to extract collision data for analysis
                    video_output_dirs = self.video_processor.setup_output_directories(
                        video_path=video_file,
                        output_root=self.processed_videos_dir,
                        model_prefix="segformer_model"
                    )
                    # Skip to collision data extraction
                    process_video = False
                else:
                    process_video = True
            else:
                process_video = True
            
            # Only do video processing if needed
            if process_video:
                # Check if we have partial processing (frames exist but no video)
                if resume and os.path.exists(frames_processed_dir):
                    frame_files = [f for f in os.listdir(frames_processed_dir) if f.startswith("frame_") and f.endswith(".png")]
                    if len(frame_files) > 0:
                        self.logger.info(f"Found {len(frame_files)} existing frames for {video_name}, attempting to complete processing...")
                        
                        # Setup directories first
                        video_output_dirs = self.video_processor.setup_output_directories(
                            video_path=video_file,
                            output_root=self.processed_videos_dir,
                            model_prefix="segformer_model"
                        )
                        
                        # Try to process with resume=True to handle partial processing
                        try:
                            video_output_dirs = self.video_processor.process_video(
                                video_path=video_file,
                                output_root=self.processed_videos_dir,
                                model_prefix="segformer_model",
                                resume=True  # Allow it to handle partial processing
                            )
                            self.logger.info(f"Successfully completed processing for {video_name}")
                        except Exception as e:
                            import traceback
                            self.logger.error(f"Failed to complete partial processing for {video_name}: {e}")
                            self.logger.error("Full traceback:")
                            for line in traceback.format_exc().splitlines():
                                self.logger.error(line)
                            self.logger.info(f"Will attempt full reprocessing for {video_name}")
                            video_output_dirs = None
                
                # If we still don't have processed video, do full processing
                if not hasattr(locals(), 'video_output_dirs') or video_output_dirs is None:
                    self.logger.info(f"Starting full processing for {video_name}")
                    try:
                        video_output_dirs = self.video_processor.process_video(
                            video_path=video_file,
                            output_root=self.processed_videos_dir,
                            model_prefix="segformer_model",
                            resume=False  # Start fresh
                        )
                    except Exception as e:
                        import traceback
                        self.logger.error(f"Failed to process video {video_name}: {e}")
                        self.logger.error("Full traceback:")
                        for line in traceback.format_exc().splitlines():
                            self.logger.error(line)
                        video_output_dirs = None
            
            # At this point, we should have video_output_dirs from either setup or processing
            if video_output_dirs is None:
                self.logger.error(f"No video output directories available for {video_name}")
                continue
            
            try:
                # ALWAYS extract collision data (either from existing results or by computing)
                collision_data_extracted = self._extract_or_compute_collision_data(
                    video_name, video_output_dirs, iou_values, process_video
                )
                
                if collision_data_extracted:
                    # Add collision data to the main dictionary
                    for iou_threshold in iou_values:
                        if (video_name, iou_threshold) not in collision_times:
                            # Load from existing file if available
                            results_dir = Path(video_output_dirs['root']) / "collision_results"
                            result_file = results_dir / f"iou_{iou_threshold}.json"
                            if result_file.exists():
                                try:
                                    with open(result_file, 'r') as f:
                                        result_data = json.load(f)
                                    collision_time = result_data.get("collision_time", float('nan'))
                                    collision_times[(video_name, iou_threshold)] = collision_time
                                    self.logger.debug(f"Loaded collision time for {video_name} at IoU {iou_threshold}: {collision_time}")
                                except Exception as e:
                                    self.logger.warning(f"Failed to load existing collision data for {video_name}, IoU {iou_threshold}: {e}")
                else:
                    self.logger.warning(f"Failed to extract collision data for video {video_name}")
                        
            except Exception as e:
                self.logger.error(f"Failed to analyze video {video_name}: {e}")
                continue
        
        return collision_times
    
    def _extract_or_compute_collision_data(self, video_name: str, video_output_dirs: Dict[str, str], 
                                         iou_values: np.ndarray, process_video: bool) -> bool:
        """
        Extract collision data either from existing results or by computing from masks.
        Returns True if successful, False otherwise.
        """
        results_dir = Path(video_output_dirs['root']) / "collision_results"
        
        # Check if collision results already exist for all IoU values
        all_exist = True
        for iou_threshold in iou_values:
            result_file = results_dir / f"iou_{iou_threshold}.json"
            if not result_file.exists():
                all_exist = False
                break
        
        if all_exist and not process_video:
            self.logger.info(f"All collision results already exist for {video_name}")
            return True
        
        # Need to compute collision data
        self.logger.info(f"Computing collision data for {video_name}")
        
        # Log blob state information (only if we processed the video)
        if process_video:
            self._log_blob_state_info(video_name, video_output_dirs)
        
        # Extract mask data from processed video
        mask_data = self._extract_mask_data_from_processed_video(video_output_dirs)
        
        if not mask_data:
            self.logger.warning(f"No mask data extracted for video {video_name}")
            return False
        
        # Create results directory
        results_dir.mkdir(exist_ok=True)
        
        # For each IoU threshold, find collision time
        for iou_threshold in iou_values:
            result_file = results_dir / f"iou_{iou_threshold}.json"
            
            # Skip if result already exists (unless we're reprocessing)
            if result_file.exists() and not process_video:
                continue
                
            collision_time = self._find_first_collision_time(mask_data, iou_threshold, fps=60)
            
            # Save individual result with blob state info
            result_data = {
                "collision_time": float(collision_time),
                "is_collision_detected": not np.isnan(collision_time),
                "iou_threshold": float(iou_threshold),
                "blob_1_disappeared": getattr(self.video_processor, 'blob_1_disappeared', False),
                "blob_1_disappeared_frame": getattr(self.video_processor, 'blob_1_disappeared_frame', None),
                "blob_1_memory_strategy": getattr(self.video_processor, 'blob_1_memory_strategy', None),
                "blob_1_memory_freeze_frame": getattr(self.video_processor, 'blob_1_memory_freeze_frame', None)
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            if not np.isnan(collision_time):
                self.logger.debug(f"Collision time for {video_name} at IoU {iou_threshold}: {collision_time:.2f}ms")
            else:
                self.logger.debug(f"No collision detected for {video_name} at IoU {iou_threshold}")
        
        return True
    
    def _collect_all_collision_data(self, all_video_names: List[str], iou_values: np.ndarray) -> Dict[Tuple[str, float], float]:
        """
        Collect collision data from all videos (both newly processed and existing).
        Returns dictionary mapping (video_name, iou_threshold) -> collision_time.
        """
        collision_data = {}
        
        for video_name in all_video_names:
            expected_output_dir = os.path.join(self.processed_videos_dir, f"segformer_model-{video_name}")
            results_dir = Path(expected_output_dir) / "collision_results"
            
            if not results_dir.exists():
                self.logger.warning(f"No collision results directory found for {video_name}")
                continue
            
            for iou_threshold in iou_values:
                result_file = results_dir / f"iou_{iou_threshold}.json"
                
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                        collision_time = result_data.get("collision_time", float('nan'))
                        collision_data[(video_name, iou_threshold)] = collision_time
                        self.logger.debug(f"Collected collision time for {video_name} at IoU {iou_threshold}: {collision_time}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load collision data for {video_name}, IoU {iou_threshold}: {e}")
                        collision_data[(video_name, iou_threshold)] = float('nan')
                else:
                    self.logger.warning(f"Missing collision result file for {video_name}, IoU {iou_threshold}")
                    collision_data[(video_name, iou_threshold)] = float('nan')
        
        return collision_data

    def _log_blob_state_info(self, video_name: str, video_output_dirs: Dict[str, str]) -> None:
        """Log blob state information for analysis."""
        blob_state_info = {
            "video_name": video_name,
            "blob_1_disappeared": self.video_processor.blob_1_disappeared,
            "blob_1_disappeared_frame": self.video_processor.blob_1_disappeared_frame,
            "blob_1_missing_threshold": self.video_processor.blob_1_missing_threshold,
            "blob_1_memory_strategy": self.video_processor.blob_1_memory_strategy,
            "blob_1_running_avg_window": self.video_processor.blob_1_running_avg_window,
            "blob_1_memory_freeze_frame": self.video_processor.blob_1_memory_freeze_frame
        }
        
        if self.video_processor.blob_1_disappeared:
            self.logger.info(f"Video {video_name}: Blob 1 disappeared at frame {self.video_processor.blob_1_disappeared_frame}")
        else:
            self.logger.info(f"Video {video_name}: Both blobs remained visible throughout")
        
        # Log memory strategy info
        self.logger.info(f"Video {video_name}: Blob 1 memory strategy = {self.video_processor.blob_1_memory_strategy}")
        if self.video_processor.blob_1_memory_freeze_frame:
            self.logger.info(f"Video {video_name}: Blob 1 memory frozen after frame {self.video_processor.blob_1_memory_freeze_frame}")
        
        # Save blob state info
        blob_state_path = Path(video_output_dirs['root']) / "blob_state_info.json"
        with open(blob_state_path, 'w') as f:
            json.dump(blob_state_info, f, indent=2)
    
    def _extract_mask_data_from_processed_video(self, video_output_dirs: Dict[str, str]) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract mask data from processed video output directories."""
        masks_dir = Path(video_output_dirs['frames_masks'])
        mask_data = {}
        
        if not masks_dir.exists():
            self.logger.warning(f"Masks directory does not exist: {masks_dir}")
            return mask_data
        
        # Find all mask files
        mask_files = list(masks_dir.glob("mask_memory_blob_*_frame_*.png"))
        
        if not mask_files:
            self.logger.warning(f"No mask files found in {masks_dir}")
            return mask_data
        
        self.logger.info(f"Found {len(mask_files)} mask files")
        
        # Group by frame number
        frame_masks = {}
        for mask_file in mask_files:
            try:
                # Parse filename: mask_memory_blob_0_frame_000013.png
                parts = mask_file.stem.split('_')
                if len(parts) < 6:
                    self.logger.warning(f"Unexpected mask filename format: {mask_file}")
                    continue
                    
                blob_idx = int(parts[3])  # blob index
                frame_num = int(parts[5])  # frame number
                
                if frame_num not in frame_masks:
                    frame_masks[frame_num] = {}
                
                # Load mask
                mask_img = Image.open(mask_file).convert('L')
                mask_array = np.array(mask_img, dtype=np.uint8)
                binary_mask = (mask_array > 0).astype(np.float32)
                
                # Only store masks that have some content
                if binary_mask.sum() > 0:
                    frame_masks[frame_num][f"blob_{blob_idx}"] = binary_mask
                
            except Exception as e:
                self.logger.warning(f"Error processing mask file {mask_file}: {e}")
                continue
        
        # Filter to only include frames with valid blob data
        # After blob 1 disappears, we only need blob 0
        valid_frame_masks = {}
        for frame_num, masks in frame_masks.items():
            if self.video_processor.blob_1_disappeared and self.video_processor.blob_1_disappeared_frame:
                # If blob 1 has disappeared, only require blob 0
                if frame_num >= self.video_processor.blob_1_disappeared_frame:
                    if "blob_0" in masks:
                        valid_frame_masks[frame_num] = masks
                else:
                    # Before blob 1 disappeared, require both blobs
                    if "blob_0" in masks and "blob_1" in masks:
                        valid_frame_masks[frame_num] = masks
            else:
                # Normal case - require both blobs
                if "blob_0" in masks and "blob_1" in masks:
                    valid_frame_masks[frame_num] = masks
        
        self.logger.info(f"Found {len(valid_frame_masks)} frames with valid blob data")
        return valid_frame_masks

    def _find_first_collision_time(self, mask_data: Dict[int, Dict[str, np.ndarray]], 
                                  iou_threshold: float, fps: int = 60) -> float:
        """Find the first frame where collision occurs based on IoU threshold."""
        
        frame_numbers = sorted(mask_data.keys())
        
        if not frame_numbers:
            self.logger.warning("No frames with mask data available")
            return float('nan')
        
        # Start checking from frame 13 (as in original code)
        start_frame = max(13, min(frame_numbers)) if frame_numbers else 13
        
        collision_found = False
        collision_frame = None
        
        for frame_num in frame_numbers:
            if frame_num < start_frame:
                continue
                
            frame_masks = mask_data[frame_num]
            
            # Check if we have both blobs for collision detection
            if "blob_0" in frame_masks and "blob_1" in frame_masks:
                mask0 = frame_masks["blob_0"]
                mask1 = frame_masks["blob_1"]
                
                # Compute IoU
                iou_val = self._compute_iou_safe(mask0, mask1)
                
                # Check if collision threshold is met
                if not np.isnan(iou_val) and iou_val >= iou_threshold:
                    collision_frame = frame_num
                    collision_found = True
                    break
            else:
                # If blob 1 has disappeared, no collision is possible
                if self.video_processor.blob_1_disappeared and frame_num >= self.video_processor.blob_1_disappeared_frame:
                    break
        
        if collision_found:
            collision_time_ms = (collision_frame / fps) * 1000
            return collision_time_ms
        else:
            return float('nan')  # No collision found

    def _compute_iou_safe(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU of two binary masks with safety checks."""
        try:
            # Ensure masks are binary
            mask1_binary = (mask1 > 0).astype(np.uint8)
            mask2_binary = (mask2 > 0).astype(np.uint8)
            
            # Compute intersection and union
            intersection = np.logical_and(mask1_binary, mask2_binary).sum()
            union = np.logical_or(mask1_binary, mask2_binary).sum()
            
            if union == 0:
                # Both masks are empty
                return 0.0
            
            iou = intersection / union
            return float(iou)
            
        except Exception as e:
            self.logger.warning(f"Error computing IoU: {e}")
            return float('nan')

    def _analyze_participant_correlations(self, collision_data: Dict[Tuple[str, float], float], 
                                        participant_df: pd.DataFrame, 
                                        iou_values: np.ndarray) -> None:
        """Analyze correlations between model predictions and participant data."""
        self.logger.info("Analyzing participant correlations...")
        
        # Get list of video names
        video_names = list(set([video_name for (video_name, _) in collision_data.keys()]))
        
        # For this analysis, we'll create a simple correlation between video names and participant stimuli
        # This assumes video file names can be mapped to participant stimulus names
        for iou_thr in iou_values:
            self._analyze_iou_threshold(iou_thr, video_names, collision_data, participant_df)

    def _analyze_iou_threshold(self, iou_thr: float, video_names: List[str],
                              collision_data: Dict[Tuple[str, float], float], 
                              participant_df: pd.DataFrame) -> None:
        """Analyze a specific IoU threshold."""
        
        # Create output directory
        out_dir_name = f"IoU_{iou_thr}"
        out_dir = Path(self.results_dir) / out_dir_name
        out_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (out_dir / "ID").mkdir(exist_ok=True)
        (out_dir / "Average_person").mkdir(exist_ok=True)
        (out_dir / "summary").mkdir(exist_ok=True)
        
        # Individual participant analysis
        self._analyze_individual_participants(
            out_dir / "ID", iou_thr, video_names, collision_data, participant_df
        )
        
        # Average participant analysis
        self._analyze_average_participant(
            out_dir / "Average_person", iou_thr, video_names, collision_data, participant_df
        )
        
        # Summary analysis
        self._generate_summary_analysis(
            out_dir / "summary", iou_thr, video_names, collision_data, participant_df
        )
        
        # Convex vs Concave analysis
        self._analyze_convex_vs_concave(
            out_dir / "convex_vs_concave", iou_thr, video_names, collision_data, participant_df
        )

    def _analyze_individual_participants(self, output_dir: Path, iou_thr: float, video_names: List[str],
                                       collision_data: Dict[Tuple[str, float], float], 
                                       participant_df: pd.DataFrame) -> None:
        """Analyze individual participant correlations."""
        
        participant_groups = participant_df.groupby("ID")
        
        for pid, group in participant_groups:
            predicted_times = []
            human_times = []
            used_videos = []
            
            for _, row in group.iterrows():
                stimulus = row.get("stimulus", "")
                # Try to match stimulus to video name (simple matching)
                video_match = self._match_stimulus_to_video(stimulus, video_names)
                
                if video_match:
                    collision_time = collision_data.get((video_match, iou_thr), float('nan'))
                    if not np.isnan(collision_time):
                        predicted_times.append(collision_time)
                        human_times.append(row["rt"])
                        used_videos.append(video_match)
            
            # Compute correlation
            r_val = self._compute_correlation(predicted_times, human_times) if len(predicted_times) > 1 else float('nan')
            
            # Create scatter plot
            fig, ax = plt.subplots()
            if predicted_times and human_times:
                ax.scatter(human_times, predicted_times, c='blue', alpha=0.6)
            ax.set_xlabel("Human RT (ms)")
            ax.set_ylabel("Model Collision Time (ms)")
            ax.set_title(f"Participant {pid}, IoU={iou_thr}, r={r_val:.3f}")
            
            # Save plot and data
            plt.savefig(output_dir / f"{pid}.png")
            plt.close(fig)
            
            with open(output_dir / f"{pid}.json", 'w') as f:
                json.dump({
                    "correlation": float(r_val) if not np.isnan(r_val) else None,
                    "videos_used": used_videos,
                    "data_points": len(predicted_times)
                }, f, indent=2)

    def _analyze_average_participant(self, output_dir: Path, iou_thr: float, video_names: List[str],
                                   collision_data: Dict[Tuple[str, float], float], 
                                   participant_df: pd.DataFrame) -> None:
        """Analyze average participant correlations."""
        
        # Map video names to average RTs
        video_rts = {name: [] for name in video_names}
        
        for _, row in participant_df.iterrows():
            stimulus = row.get("stimulus", "")
            video_match = self._match_stimulus_to_video(stimulus, video_names)
            if video_match and video_match in video_rts:
                video_rts[video_match].append(row["rt"])
        
        # Compute averages and correlations
        avg_human_times = []
        model_times = []
        used_videos = []
        
        for video_name in video_names:
            rts = video_rts[video_name]
            if len(rts) > 0:
                avg_rt = np.mean(rts)
                collision_time = collision_data.get((video_name, iou_thr), float('nan'))
                if not np.isnan(collision_time):
                    avg_human_times.append(avg_rt)
                    model_times.append(collision_time)
                    used_videos.append(video_name)
        
        # Compute correlation
        r_val = self._compute_correlation(avg_human_times, model_times) if len(avg_human_times) > 1 else float('nan')
        
        # Create plot
        fig, ax = plt.subplots()
        if avg_human_times and model_times:
            ax.scatter(avg_human_times, model_times, c='red', alpha=0.7)
        ax.set_xlabel("Average Human RT (ms)")
        ax.set_ylabel("Model Collision Time (ms)")
        ax.set_title(f"Average Person, IoU={iou_thr}, r={r_val:.3f}")
        
        plt.savefig(output_dir / "average_person.png")
        plt.close(fig)
        
        with open(output_dir / "average_person.json", 'w') as f:
            json.dump({
                "correlation": float(r_val) if not np.isnan(r_val) else None,
                "videos_used": used_videos,
                "data_points": len(avg_human_times)
            }, f, indent=2)

    def _generate_summary_analysis(self, output_dir: Path, iou_thr: float, video_names: List[str],
                                 collision_data: Dict[Tuple[str, float], float], 
                                 participant_df: pd.DataFrame) -> None:
        """Generate summary analysis for this IoU threshold."""
        
        # Collect collision times for all videos at this IoU threshold
        video_collision_times = {}
        collision_count = 0
        
        for video_name in video_names:
            collision_time = collision_data.get((video_name, iou_thr), float('nan'))
            video_collision_times[video_name] = collision_time
            if not np.isnan(collision_time):
                collision_count += 1
        
        # Create summary plot
        if video_collision_times:
            names = list(video_collision_times.keys())
            times = list(video_collision_times.values())
            
            # Filter out NaN values for plotting
            valid_indices = [i for i, t in enumerate(times) if not np.isnan(t)]
            valid_names = [names[i] for i in valid_indices]
            valid_times = [times[i] for i in valid_indices]
            
            if valid_times:
                fig, ax = plt.subplots(figsize=(15, 6))
                bars = ax.bar(range(len(valid_names)), valid_times, color='steelblue', alpha=0.7)
                ax.set_xlabel('Video Name')
                ax.set_ylabel('Collision Time (ms)')
                ax.set_title(f'Collision Times for IoU Threshold {iou_thr} ({collision_count}/{len(names)} collisions detected)')
                ax.set_xticks(range(len(valid_names)))
                ax.set_xticklabels(valid_names, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, time in zip(bars, valid_times):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(valid_times)*0.01,
                           f'{time:.1f}', ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                plt.savefig(output_dir / f"collision_times_iou_{iou_thr}.png", dpi=300)
                plt.close(fig)
        
        # Save summary data
        summary_data = {
            "iou_threshold": float(iou_thr),
            "video_collision_times": {k: float(v) if not np.isnan(v) else None for k, v in video_collision_times.items()},
            "total_videos": len(video_names),
            "videos_with_collisions": collision_count,
            "collision_rate": collision_count / len(video_names) if video_names else 0.0
        }
        
        with open(output_dir / f"summary_iou_{iou_thr}.json", 'w') as f:
            json.dump(summary_data, f, indent=2)

    def _match_stimulus_to_video(self, stimulus: str, video_names: List[str]) -> str:
        """
        Match participant stimulus name to video file name.
        This is a simple matching function that can be customized based on naming conventions.
        """
        if not stimulus:
            return None
        
        # Simple matching: look for video names that contain the stimulus or vice versa
        stimulus_clean = stimulus.lower().replace('stimulus/', '').replace('.mp4', '')
        
        for video_name in video_names:
            video_clean = video_name.lower()
            if stimulus_clean in video_clean or video_clean in stimulus_clean:
                return video_name
        
        return None

    def _parse_video_name(self, video_name: str) -> Dict[str, Any]:
        """
        Parse video name to extract concave/convex information and ground‑truth time.

        Expected format: e.g. "BConcave+AConcave+3500" or "BConvex+AConvex+2000".
        The overall interface (return keys) is preserved, but `is_concave`
        now depends *only* on whether the **second** token contains
        "concave" or "convex".
        """
        import re

        # Remove common video extensions
        base_name = video_name.replace('.mp4', '').replace('.avi', '')

        # Split by '+' to get components
        parts = base_name.split('+')

        # Early exit if format is unexpected
        if len(parts) < 3:
            return {
                "is_concave": None,
                "ground_truth": None,
                "tokens": parts,
                "concave_count": 0,
                "convex_count": 0
            }

        # --------------------------- ground‑truth ---------------------------
        ground_truth = np.nan
        for part in reversed(parts):
            nums = re.findall(r'\d+', part)
            if nums:
                ground_truth = int(nums[-1])
                break

        # --------------------------- concave/convex counts -----------------
        concave_count = sum('concave' in p.lower() for p in parts)
        convex_count = sum('convex' in p.lower() for p in parts)

        # --------------------------- is_concave decision -------------------
        # Only the *second* token (index 1) drives the classification
        second_token = parts[1].lower() if len(parts) > 1 else ""
        if 'concave' in second_token:
            is_concave = True
        elif 'convex' in second_token:
            is_concave = False
        else:
            is_concave = None  # Unknown / undecidable from second token

        return {
            "is_concave": is_concave,
            "ground_truth": ground_truth,
            "tokens": parts,
            "concave_count": concave_count,
            "convex_count": convex_count
        }

    def _is_concave_token(self, token: str) -> bool:
        """Check if a token represents concave shape."""
        return "concave" in token.lower()
    
    def _analyze_convex_vs_concave(self, output_dir: Path, iou_thr: float, video_names: List[str],
                                 collision_data: Dict[Tuple[str, float], float], 
                                 participant_df: pd.DataFrame) -> None:
        """
        Analyze differences between concave and convex videos, similar to original data_analysis_v2.py.
        """
        output_dir.mkdir(exist_ok=True)
        
        # Parse all video names to get concave/convex info
        video_info = {}
        for video_name in video_names:
            parsed = self._parse_video_name(video_name)
            if parsed['is_concave'] is not None and parsed['ground_truth'] is not None:
                video_info[video_name] = parsed
        
        if not video_info:
            self.logger.warning(f"No videos with valid concave/convex info for IoU {iou_thr}")
            return
        
        # Group by ground truth time
        gt_to_concave_vals = {}
        gt_to_convex_vals = {}
        
        # Model data: group collision times by ground truth and concave/convex
        for video_name, info in video_info.items():
            gt = info['ground_truth']
            is_concave = info['is_concave']
            collision_time = collision_data.get((video_name, iou_thr), float('nan'))
            
            if not np.isnan(collision_time):
                if is_concave:
                    gt_to_concave_vals.setdefault(gt, []).append(collision_time)
                else:
                    gt_to_convex_vals.setdefault(gt, []).append(collision_time)
        
        # Human data: group by ground truth and is_concave from CSV
        if 'groundTruth' in participant_df.columns and 'is_concave' in participant_df.columns:
            df_grouped = participant_df.groupby(['groundTruth', 'is_concave'])['rt'].mean().reset_index()
            
            # Convert to the same format as model data
            human_gt_to_concave = {}
            human_gt_to_convex = {}
            
            for _, row in df_grouped.iterrows():
                gt = row['groundTruth']
                is_concave = bool(row['is_concave'])
                avg_rt = row['rt']
                
                if is_concave:
                    human_gt_to_concave.setdefault(gt, []).append(avg_rt)
                else:
                    human_gt_to_convex.setdefault(gt, []).append(avg_rt)
        else:
            self.logger.warning("No groundTruth or is_concave columns in participant data")
            human_gt_to_concave = {}
            human_gt_to_convex = {}
        
        # Compute model means and standard errors for each ground truth time
        gt_sorted = sorted(set(list(gt_to_concave_vals.keys()) + list(gt_to_convex_vals.keys())))
        
        # Model data: compute means and standard errors for concave vs convex
        model_concave_means = []
        model_concave_sems = []
        model_convex_means = []
        model_convex_sems = []
        
        for gt in gt_sorted:
            # Model concave data
            concave_times = gt_to_concave_vals.get(gt, [])
            if concave_times:
                concave_mean = np.mean(concave_times)
                concave_sem = np.std(concave_times, ddof=1) / np.sqrt(len(concave_times)) if len(concave_times) > 1 else 0
            else:
                concave_mean = float('nan')
                concave_sem = 0
            model_concave_means.append(concave_mean)
            model_concave_sems.append(concave_sem)
            
            # Model convex data
            convex_times = gt_to_convex_vals.get(gt, [])
            if convex_times:
                convex_mean = np.mean(convex_times)
                convex_sem = np.std(convex_times, ddof=1) / np.sqrt(len(convex_times)) if len(convex_times) > 1 else 0
            else:
                convex_mean = float('nan')
                convex_sem = 0
            model_convex_means.append(convex_mean)
            model_convex_sems.append(convex_sem)
        
        # Create two-column bar chart visualization using same style as gen_two_box_plots.py
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define parameters for the grouped bar chart (same as gen_two_box_plots.py)
        bar_width = 0.40
        x_positions = np.arange(len(gt_sorted)) * 1.0
        
        # Place bars directly next to each other using exact same colors as gen_two_box_plots.py
        concave_bars = ax.bar(x_positions - bar_width / 2, model_concave_means, bar_width,
                             color='#FFBE48', label='Concave')
        convex_bars = ax.bar(x_positions + bar_width / 2, model_convex_means, bar_width,
                            color='#56A036', label='Convex')
        
        # Add error bars on TOP of each bar (same as gen_two_box_plots.py)
        ax.errorbar(x_positions - bar_width / 2, model_concave_means, yerr=model_concave_sems,
                   fmt='none', ecolor='black', capsize=3)
        ax.errorbar(x_positions + bar_width / 2, model_convex_means, yerr=model_convex_sems,
                   fmt='none', ecolor='black', capsize=3)
        
        # Set axis labels and title
        ax.set_xlabel('Ground Truth Time-to-Collision (ms)')
        ax.set_ylabel('Model Time-to-Collision (ms)')
        ax.set_title(f'Concave vs Convex Collision Times, IoU={iou_thr}')
        
        # Customize x-axis
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(g) for g in gt_sorted])
        
        # Add a grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / "concave_vs_convex.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save data
        analysis_data = {
            "iou_threshold": float(iou_thr),
            "ground_truth_times": gt_sorted,
            "model_concave_means": [float(m) if not np.isnan(m) else None for m in model_concave_means],
            "model_concave_sems": [float(s) for s in model_concave_sems],
            "model_convex_means": [float(m) if not np.isnan(m) else None for m in model_convex_means],
            "model_convex_sems": [float(s) for s in model_convex_sems],
            "video_info": {k: v for k, v in video_info.items()},
            "summary": {
                "total_ground_truth_times": len(gt_sorted),
                "valid_concave_data_points": sum(1 for m in model_concave_means if not np.isnan(m)),
                "valid_convex_data_points": sum(1 for m in model_convex_means if not np.isnan(m))
            }
        }
        
        with open(output_dir / "concave_vs_convex_analysis.json", 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        self.logger.info(f"Convex vs Concave analysis completed for IoU {iou_thr}")

    def _compute_correlation(self, xvals: List[float], yvals: List[float]) -> float:
        """
        Return Pearson correlation. If insufficient data, return NaN.
        Includes safeguards against numpy runtime warnings.
        """
        if len(xvals) < 2 or len(yvals) < 2:
            return float('nan')
        
        # Convert to numpy arrays for easier handling
        x_arr = np.array(xvals)
        y_arr = np.array(yvals)
        
        # Check for NaN or infinite values
        if np.any(np.isnan(x_arr)) or np.any(np.isnan(y_arr)):
            return float('nan')
        if np.any(np.isinf(x_arr)) or np.any(np.isinf(y_arr)):
            return float('nan')
        
        # Check for zero variance (constant values)
        if np.std(x_arr) == 0 or np.std(y_arr) == 0:
            return float('nan')
        
        try:
            # Compute correlation with proper error handling
            correlation_matrix = np.corrcoef(x_arr, y_arr)
            r = correlation_matrix[0, 1]
            return float(r) if not np.isnan(r) else float('nan')
        except Exception as e:
            self.logger.warning(f"Error computing correlation: {e}")
            return float('nan')

##############################################################################
# MAIN FUNCTION
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Time-to-Collision (TTC) Experiment - Process raw videos and correlate with human response times")
    parser.add_argument("--model_interface", type=str, default="segformer",
                      choices=["segformer"], help="Model interface to use")
    parser.add_argument("--videos_dir", type=str, required=False,
                      default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/hugging_face/model_experiments/exp2TTC_files",
                      help="Directory containing raw .mp4 video files")
    parser.add_argument("--csv_path", type=str, required=False,
                      default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/hugging_face/model_experiments/exp2TTC_files/experiment2-CollisionDetection-Data.csv",
                      help="Path to CSV file with participant data")
    parser.add_argument("--output_dir", type=str, required=False,
                      default="/home/projects/bagon/andreyg/Projects/Object_reps_neural/Programming/hugging_face/model_experiments/segformer/exp2TTC",
                      help="Output directory for results and processed data")
    parser.add_argument("--n_blobs", type=int, default=2,
                      help="Number of blobs to detect and track (default: 2)")
    parser.add_argument("--blob_1_memory_freeze_frame", type=int, default=80,
                      help="Frame after which blob 1 memory stops updating (default: 80)")
    parser.add_argument("--resume", action="store_true", default=True,
                      help="Resume processing from checkpoints (default: True)")
    parser.add_argument("--no_resume", action="store_true", default=False,
                      help="Start processing from scratch, ignoring checkpoints")
    
    args = parser.parse_args()
    
    # Handle resume logic
    resume = args.resume and not args.no_resume
    
    # Validate inputs
    if not os.path.isdir(args.videos_dir):
        print(f"Error: Videos directory '{args.videos_dir}' does not exist")
        sys.exit(1)
    
    if not os.path.isfile(args.csv_path):
        print(f"Error: CSV file '{args.csv_path}' does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model interface
    if args.model_interface == "segformer":
        model_interface = SegFormerInterface()
    else:
        raise ValueError(f"Unknown model interface: {args.model_interface}")
    
    # Run experiment
    experiment = TTCExperiment(
        model_interface, 
        args.output_dir, 
        args.n_blobs,
        blob_1_memory_freeze_frame=args.blob_1_memory_freeze_frame
    )
    
    try:
        experiment.run_full_experiment(
            videos_dir=args.videos_dir,
            csv_path=args.csv_path,
            resume=resume
        )
        print("TTC experiment completed successfully!")
        
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