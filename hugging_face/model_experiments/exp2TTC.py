#!/usr/bin/env python3
"""
exp2TTC.py

Time-to-Collision (TTC) Experiment that processes raw .mp4 videos and computes collision detection
times under varying IoU thresholds, correlating with participant response data.
Completely self-contained from raw videos to final analysis.

Usage:
    python exp2TTC.py --model_interface segformer --videos_dir /path/to/raw_videos --csv_path /path/to/participants.csv --output_dir /path/to/output [--resume]
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
    """

    def __init__(self, model_interface: ModelInterface, output_dir: str, n_blobs: int = 2,
                 logger: logging.Logger = None):
        self.model_interface = model_interface
        self.output_dir = output_dir

        # Create output subdirectories FIRST (before logger is set up)
        self.results_dir = os.path.join(output_dir, "results")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.processed_videos_dir = os.path.join(output_dir, "processed_videos")

        for dir_path in [self.results_dir, self.plots_dir, self.logs_dir, self.processed_videos_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Now setup logger after logs_dir exists
        self.logger = logger or self._setup_logger()

        # Initialize video processor
        self.video_processor = VideoProcessor(model_interface, n_blobs, self.logger)

        self.logger.info(f"Initialized TTC Experiment with output dir: {output_dir}")

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
        
        # Step 4: Analyze correlations with participant data
        if collision_data:
            self._analyze_participant_correlations(collision_data, participant_df, iou_values)
        else:
            self.logger.warning("No collision data generated - skipping correlation analysis")
        
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
        
        for video_file in video_files:
            video_name = Path(video_file).stem
            self.logger.info(f"Processing video: {video_name}")
            
            # Use VideoProcessor to process the entire video
            try:
                video_output_dirs = self.video_processor.process_video(
                    video_path=video_file,
                    output_root=self.processed_videos_dir,
                    model_prefix="segformer_model",
                    resume=resume
                )
                
                # Extract mask data from processed video
                mask_data = self._extract_mask_data_from_processed_video(video_output_dirs)
                
                # For each IoU threshold, find collision time
                for iou_threshold in iou_values:
                    collision_time = self._find_first_collision_time(mask_data, iou_threshold, fps=60)
                    collision_times[(video_name, iou_threshold)] = collision_time
                    
                    # Save individual result
                    results_dir = Path(video_output_dirs['root']) / "collision_results"
                    results_dir.mkdir(exist_ok=True)
                    output_json_path = results_dir / f"iou_{iou_threshold}.json"
                    with open(output_json_path, 'w') as f:
                        json.dump({"collision_time": collision_time}, f, indent=2)
                    
                    self.logger.debug(f"Collision time for {video_name} at IoU {iou_threshold}: {collision_time}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process video {video_name}: {e}")
                continue
        
        return collision_times
    
    def _extract_mask_data_from_processed_video(self, video_output_dirs: Dict[str, str]) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract mask data from processed video output directories."""
        masks_dir = Path(video_output_dirs['frames_masks_nonmem'])
        mask_data = {}
        
        # Find all mask files
        mask_files = list(masks_dir.glob("mask_blob_*_frame_*.png"))
        
        # Group by frame number
        frame_masks = {}
        for mask_file in mask_files:
            # Parse filename: mask_blob_0_frame_000013.png
            parts = mask_file.stem.split('_')
            blob_idx = int(parts[2])  # blob index
            frame_num = int(parts[4])  # frame number
            
            if frame_num not in frame_masks:
                frame_masks[frame_num] = {}
            
            # Load mask
            mask_img = Image.open(mask_file).convert('L')
            mask_array = np.array(mask_img, dtype=np.uint8)
            binary_mask = (mask_array > 0).astype(np.float32)
            
            frame_masks[frame_num][f"blob_{blob_idx}"] = binary_mask
        
        return frame_masks

    def _find_first_collision_time(self, mask_data: Dict[int, Dict[str, np.ndarray]], 
                                  iou_threshold: float, fps: int = 60) -> float:
        """Find the first frame where collision occurs based on IoU threshold."""
        
        frame_numbers = sorted(mask_data.keys())
        start_frame = max(13, min(frame_numbers)) if frame_numbers else 13
        
        for frame_num in frame_numbers:
            if frame_num < start_frame:
                continue
                
            frame_masks = mask_data[frame_num]
            if "blob_0" in frame_masks and "blob_1" in frame_masks:
                mask0 = frame_masks["blob_0"]
                mask1 = frame_masks["blob_1"]
                
                iou_val = self._compute_iou(mask0, mask1)
                if iou_val >= iou_threshold:
                    collision_time_ms = (frame_num / fps) * 1000
                    return collision_time_ms
        
        return float('nan')  # No collision found

    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU of two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union

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
                    "correlation": r_val,
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
                "correlation": r_val,
                "videos_used": used_videos,
                "data_points": len(avg_human_times)
            }, f, indent=2)

    def _generate_summary_analysis(self, output_dir: Path, iou_thr: float, video_names: List[str],
                                 collision_data: Dict[Tuple[str, float], float], 
                                 participant_df: pd.DataFrame) -> None:
        """Generate summary analysis for this IoU threshold."""
        
        # Collect collision times for all videos at this IoU threshold
        video_collision_times = {}
        for video_name in video_names:
            collision_time = collision_data.get((video_name, iou_thr), float('nan'))
            if not np.isnan(collision_time):
                video_collision_times[video_name] = collision_time
        
        # Create summary plot
        if video_collision_times:
            names = list(video_collision_times.keys())
            times = list(video_collision_times.values())
            
            fig, ax = plt.subplots(figsize=(15, 6))
            bars = ax.bar(range(len(names)), times, color='steelblue', alpha=0.7)
            ax.set_xlabel('Video Name')
            ax.set_ylabel('Collision Time (ms)')
            ax.set_title(f'Collision Times for IoU Threshold {iou_thr}')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, time in zip(bars, times):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                       f'{time:.1f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"collision_times_iou_{iou_thr}.png")
            plt.close(fig)
        
        # Save summary data
        summary_data = {
            "iou_threshold": iou_thr,
            "video_collision_times": video_collision_times,
            "total_videos": len(video_names),
            "videos_with_collisions": len(video_collision_times)
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

    def _compute_correlation(self, xvals: List[float], yvals: List[float]) -> float:
        """
        Return Pearson correlation. If insufficient data, return NaN.
        """
        if len(xvals) < 2 or len(yvals) < 2:
            return float('nan')
        try:
            r = np.corrcoef(xvals, yvals)[0, 1]
            return float(r) if not np.isnan(r) else float('nan')
        except:
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
    parser.add_argument("--resume", action="store_true", default=False,
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
    experiment = TTCExperiment(model_interface, args.output_dir, args.n_blobs)
    
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
        print(f"Experiment failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 