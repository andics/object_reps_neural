#!/usr/bin/env python3
"""
exp1Causality.py

Causality Experiment that processes raw .mp4 videos and computes causality scores
based on object distance changes over time. Completely self-contained from raw videos to final analysis.

Usage:
    python exp1Causality.py --model_interface segformer --videos_dir /path/to/raw_videos --output_dir /path/to/output [--resume]
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

class CausalityExperiment:
    """
    Experiment 1: Causality Analysis
    
    This experiment:
    1. Takes a directory of raw .mp4 video files as input
    2. Processes each video using VideoProcessor to extract frames and detect objects
    3. Computes distance changes between objects over time
    4. Calculates causality scores based on distance patterns
    5. Generates analysis plots and saves results
    """
    
    def __init__(self, model_interface: ModelInterface, output_dir: str, n_blobs: int = 2, logger: logging.Logger = None):
        self.model_interface = model_interface
        self.output_dir = output_dir
        
        # Create output subdirectories FIRST (before logger is set up)
        self.results_dir = os.path.join(output_dir, "results")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.processed_videos_dir = os.path.join(output_dir, "processed_videos")
        self.distance_data_dir = os.path.join(output_dir, "distance_data")
        
        for dir_path in [self.results_dir, self.plots_dir, self.logs_dir, 
                        self.processed_videos_dir, self.distance_data_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Now setup logger after logs_dir exists
        self.logger = logger or self._setup_logger()
        
        # Initialize video processor
        self.video_processor = VideoProcessor(model_interface, n_blobs, self.logger)
            
        self.logger.info(f"Initialized Causality Experiment with output dir: {output_dir}")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(self.logs_dir, f"causality_exp_{timestamp}.log")

        logger = logging.getLogger(f"causality_exp_{timestamp}")
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

    def run_full_experiment(self, videos_dir: str, resume: bool = True) -> None:
        """Run the complete causality experiment from raw videos to final analysis."""
        self.logger.info("Starting full causality experiment")
        
        # Step 1: Find all .mp4 video files in the input directory
        video_files = self._find_video_files(videos_dir)
        if not video_files:
            self.logger.error(f"No .mp4 video files found in {videos_dir}")
            return
        
        self.logger.info(f"Found {len(video_files)} video files to process")
        
        # Step 2: Process each video to extract frames and detect objects
        all_distance_data = {}
        
        for video_file in video_files:
            video_name = Path(video_file).stem
            self.logger.info(f"Processing video: {video_name}")
            
            try:
                # Process video using VideoProcessor
                video_output_dirs = self.video_processor.process_video(
                    video_path=video_file,
                    output_root=self.processed_videos_dir,
                    model_prefix="segformer_model",
                    resume=resume
                )
                
                # Log blob state information
                self._log_blob_state_info(video_name, video_output_dirs)
                
                # Extract distance data from processed video
                distance_data = self._extract_distance_data_from_processed_video(
                    video_output_dirs, video_name
                )
                
                if distance_data:
                    all_distance_data[video_name] = distance_data
                    
                    # Save individual distance data with blob state info
                    distance_csv_path = os.path.join(self.distance_data_dir, f"{video_name}_distances.csv")
                    self._save_distance_data_to_csv(distance_data, distance_csv_path)
                    
                else:
                    self.logger.warning(f"No distance data extracted for video: {video_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process video {video_name}: {e}")
                continue
        
        # Step 3: Compute causality scores for all videos
        if all_distance_data:
            self._compute_and_analyze_causality_scores(all_distance_data)
        else:
            self.logger.warning("No distance data available - skipping causality analysis")
        
        self.logger.info("Causality experiment completed successfully")

    def _find_video_files(self, videos_dir: str) -> List[str]:
        """Find all .mp4 video files in the specified directory."""
        video_pattern = os.path.join(videos_dir, "*.mp4")
        video_files = glob.glob(video_pattern)
        return sorted(video_files)

    def _log_blob_state_info(self, video_name: str, video_output_dirs: Dict[str, str]) -> None:
        """Log blob state information for analysis."""
        blob_state_info = {
            "video_name": video_name,
            "blob_1_disappeared": self.video_processor.blob_1_disappeared,
            "blob_1_disappeared_frame": self.video_processor.blob_1_disappeared_frame,
            "blob_1_missing_threshold": self.video_processor.blob_1_missing_threshold
        }
        
        if self.video_processor.blob_1_disappeared:
            self.logger.info(f"Video {video_name}: Blob 1 disappeared at frame {self.video_processor.blob_1_disappeared_frame}")
        else:
            self.logger.info(f"Video {video_name}: Both blobs remained visible throughout")
        
        # Save blob state info
        blob_state_path = Path(video_output_dirs['root']) / "blob_state_info.json"
        with open(blob_state_path, 'w') as f:
            json.dump(blob_state_info, f, indent=2)

    def _extract_distance_data_from_processed_video(self, video_output_dirs: Dict[str, str], 
                                                   video_name: str) -> List[Dict[str, Any]]:
        """Extract distance data from processed video output directories."""
        masks_dir = Path(video_output_dirs['frames_masks_nonmem'])
        
        if not masks_dir.exists():
            self.logger.warning(f"Masks directory does not exist: {masks_dir}")
            return []
        
        # Find all mask files and group by frame
        mask_files = list(masks_dir.glob("mask_blob_*_frame_*.png"))
        
        if not mask_files:
            self.logger.warning(f"No mask files found in {masks_dir}")
            return []
        
        self.logger.info(f"Found {len(mask_files)} mask files for video {video_name}")
        
        # Group masks by frame number
        frame_masks = {}
        for mask_file in mask_files:
            try:
                # Parse filename: mask_blob_0_frame_000013.png
                parts = mask_file.stem.split('_')
                if len(parts) < 5:
                    self.logger.warning(f"Unexpected mask filename format: {mask_file}")
                    continue
                    
                blob_idx = int(parts[2])  # blob index
                frame_num = int(parts[4])  # frame number
                
                if frame_num not in frame_masks:
                    frame_masks[frame_num] = {}
                
                # Load mask and compute centroid
                mask_img = Image.open(mask_file).convert('L')
                mask_array = np.array(mask_img, dtype=np.uint8)
                binary_mask = (mask_array > 0).astype(np.float32)
                
                # Only process masks with some content
                if binary_mask.sum() > 0:
                    # Compute centroid
                    centroid = self._compute_mask_centroid(binary_mask)
                    frame_masks[frame_num][f"blob_{blob_idx}"] = {
                        'mask': binary_mask,
                        'centroid': centroid
                    }
                    
            except Exception as e:
                self.logger.warning(f"Error processing mask file {mask_file}: {e}")
                continue
        
        # Convert to distance data format, considering blob state
        distance_data = []
        frame_numbers = sorted(frame_masks.keys())
        
        valid_frame_count = 0
        for frame_num in frame_numbers:
            frame_data = frame_masks[frame_num]
            
            # Check if we should compute distances based on blob state
            if self.video_processor.blob_1_disappeared and self.video_processor.blob_1_disappeared_frame:
                # If blob 1 has disappeared, only process frames before disappearance
                if frame_num >= self.video_processor.blob_1_disappeared_frame:
                    break
            
            # Need at least 2 blobs to compute distance
            if len(frame_data) >= 2:
                blob_0 = frame_data.get('blob_0')
                blob_1 = frame_data.get('blob_1')
                
                if blob_0 and blob_1:
                    # Compute distance between centroids
                    distance = self._compute_distance(blob_0['centroid'], blob_1['centroid'])
                    
                    distance_data.append({
                        'frame': frame_num,
                        'distance': distance,
                        'blob_0_centroid': blob_0['centroid'],
                        'blob_1_centroid': blob_1['centroid']
                    })
                    valid_frame_count += 1
        
        self.logger.info(f"Extracted {len(distance_data)} distance measurements from {valid_frame_count} valid frames for {video_name}")
        return distance_data

    def _compute_mask_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """Compute centroid of a binary mask."""
        if mask.sum() == 0:
            return (0.0, 0.0)
        
        y_coords, x_coords = np.where(mask > 0)
        centroid_x = float(np.mean(x_coords))
        centroid_y = float(np.mean(y_coords))
        return (centroid_x, centroid_y)

    def _compute_distance(self, centroid1: Tuple[float, float], 
                         centroid2: Tuple[float, float]) -> float:
        """Compute Euclidean distance between two centroids."""
        dx = centroid1[0] - centroid2[0]
        dy = centroid1[1] - centroid2[1]
        return float(np.sqrt(dx**2 + dy**2))

    def _save_distance_data_to_csv(self, distance_data: List[Dict[str, Any]], 
                                  csv_path: str) -> None:
        """Save distance data to CSV file."""
        if not distance_data:
            return
        
        df_data = []
        for entry in distance_data:
            df_data.append({
                'frame': entry['frame'],
                'distance': entry['distance'],
                'blob_0_x': entry['blob_0_centroid'][0],
                'blob_0_y': entry['blob_0_centroid'][1],
                'blob_1_x': entry['blob_1_centroid'][0],
                'blob_1_y': entry['blob_1_centroid'][1]
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved distance data to {csv_path}")

    def _compute_and_analyze_causality_scores(self, all_distance_data: Dict[str, List[Dict[str, Any]]]) -> None:
        """Compute causality scores for all videos and generate analysis plots."""
        self.logger.info("Computing causality scores for all videos")
        
        causality_results = {}
        
        for video_name, distance_data in all_distance_data.items():
            if len(distance_data) < 20:  # Need sufficient data points
                self.logger.warning(f"Insufficient data points for {video_name}: {len(distance_data)}")
                continue
            
            # Extract distance values and frames
            frames = [entry['frame'] for entry in distance_data]
            distances = [entry['distance'] for entry in distance_data]
            
            # Compute causality score based on distance changes
            causality_score = self._compute_causality_score(frames, distances)
            
            # Add blob state information to results
            causality_results[video_name] = {
                'causality_score': causality_score,
                'frame_count': len(distance_data),
                'distance_data': distance_data,
                'blob_1_disappeared': self.video_processor.blob_1_disappeared,
                'blob_1_disappeared_frame': self.video_processor.blob_1_disappeared_frame
            }
            
            # Generate individual plot for this video
            self._generate_causality_plot(video_name, frames, distances, causality_score)
            
            self.logger.info(f"Video {video_name}: causality score = {causality_score:.4f}")
        
        # Save summary results
        self._save_causality_summary(causality_results)
        
        # Generate comparative analysis plots
        self._generate_comparative_analysis(causality_results)

    def _compute_causality_score(self, frames: List[int], distances: List[float]) -> float:
        """
        Compute causality score based on distance change patterns.
        
        Causality score is computed as the rate of distance change over time,
        with higher scores indicating stronger causal interaction.
        """
        if len(distances) < 2:
            return 0.0
        
        # Compute distance differences (velocity)
        distance_diffs = np.diff(distances)
        
        # Apply shift logic - skip initial frames (similar to original implementation)
        shift_frames = 13  # Skip first 13 frames as in original
        if len(distance_diffs) <= shift_frames:
            return 0.0
        
        # Use distance differences after the shift
        relevant_diffs = distance_diffs[shift_frames:]
        
        # Compute causality score as mean absolute change rate
        if len(relevant_diffs) > 0:
            causality_score = float(np.mean(np.abs(relevant_diffs)))
        else:
            causality_score = 0.0
        
        return causality_score

    def _generate_causality_plot(self, video_name: str, frames: List[int], 
                               distances: List[float], causality_score: float) -> None:
        """Generate a causality plot for individual video."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Distance over time
        ax1.plot(frames, distances, 'b-', linewidth=2, label='Distance')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Distance (pixels)')
        ax1.set_title(f'{video_name} - Distance Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add marker for blob 1 disappearance if applicable
        if self.video_processor.blob_1_disappeared and self.video_processor.blob_1_disappeared_frame:
            ax1.axvline(x=self.video_processor.blob_1_disappeared_frame, color='red', 
                       linestyle='--', alpha=0.7, label=f'Blob 1 disappeared (frame {self.video_processor.blob_1_disappeared_frame})')
            ax1.legend()
        
        # Plot 2: Distance differences (velocity)
        if len(distances) > 1:
            distance_diffs = np.diff(distances)
            diff_frames = frames[1:]
            ax2.plot(diff_frames, distance_diffs, 'r-', linewidth=2, label='Distance Change')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Frame Number')
            ax2.set_ylabel('Distance Change (pixels/frame)')
            ax2.set_title(f'{video_name} - Distance Change Rate (Causality Score: {causality_score:.4f})')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add marker for blob 1 disappearance if applicable
            if self.video_processor.blob_1_disappeared and self.video_processor.blob_1_disappeared_frame:
                ax2.axvline(x=self.video_processor.blob_1_disappeared_frame, color='red', 
                           linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f"{video_name}_causality.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Saved causality plot to {plot_path}")

    def _save_causality_summary(self, causality_results: Dict[str, Dict[str, Any]]) -> None:
        """Save summary of causality results to JSON and CSV."""
        
        # Save detailed JSON
        json_path = os.path.join(self.results_dir, "causality_results.json")
        
        # Prepare JSON data (exclude raw distance data for cleaner output)
        json_data = {}
        for video_name, results in causality_results.items():
            json_data[video_name] = {
                'causality_score': float(results['causality_score']),
                'frame_count': int(results['frame_count']),
                'blob_1_disappeared': results.get('blob_1_disappeared', False),
                'blob_1_disappeared_frame': results.get('blob_1_disappeared_frame', None)
            }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save CSV summary
        csv_path = os.path.join(self.results_dir, "causality_summary.csv")
        
        summary_data = []
        for video_name, results in causality_results.items():
            summary_data.append({
                'video_name': video_name,
                'causality_score': results['causality_score'],
                'frame_count': results['frame_count'],
                'blob_1_disappeared': results.get('blob_1_disappeared', False),
                'blob_1_disappeared_frame': results.get('blob_1_disappeared_frame', None)
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('causality_score', ascending=False)
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Saved causality summary to {json_path} and {csv_path}")

    def _generate_comparative_analysis(self, causality_results: Dict[str, Dict[str, Any]]) -> None:
        """Generate comparative analysis plots across all videos."""
        
        # Extract video names and scores
        video_names = list(causality_results.keys())
        causality_scores = [results['causality_score'] for results in causality_results.values()]
        blob_1_disappeared = [results.get('blob_1_disappeared', False) for results in causality_results.values()]
        
        if not video_names:
            return
        
        # Sort by causality score for better visualization
        sorted_indices = np.argsort(causality_scores)[::-1]  # Descending order
        sorted_names = [video_names[i] for i in sorted_indices]
        sorted_scores = [causality_scores[i] for i in sorted_indices]
        sorted_blob_disappeared = [blob_1_disappeared[i] for i in sorted_indices]
        
        # Create bar plot with color coding for blob disappearance
        fig, ax = plt.subplots(figsize=(15, 8))
        
        colors = ['red' if disappeared else 'steelblue' for disappeared in sorted_blob_disappeared]
        bars = ax.bar(range(len(sorted_names)), sorted_scores, 
                     color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Video Name')
        ax.set_ylabel('Causality Score')
        ax.set_title('Causality Scores Across All Videos\n(Red: Blob 1 Disappeared, Blue: Both Blobs Visible)')
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sorted_scores)*0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        comparison_path = os.path.join(self.plots_dir, "causality_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Generate histogram of causality scores
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(sorted_scores, bins=min(20, len(sorted_scores)), color='lightblue', 
               edgecolor='black', alpha=0.7)
        ax.set_xlabel('Causality Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Causality Scores')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_score = np.mean(sorted_scores)
        std_score = np.std(sorted_scores)
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_score:.3f}')
        ax.axvline(mean_score + std_score, color='orange', linestyle='--', linewidth=1,
                  label=f'Mean + Std: {mean_score + std_score:.3f}')
        ax.axvline(mean_score - std_score, color='orange', linestyle='--', linewidth=1,
                  label=f'Mean - Std: {mean_score - std_score:.3f}')
        ax.legend()
        
        plt.tight_layout()
        histogram_path = os.path.join(self.plots_dir, "causality_distribution.png")
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.logger.info(f"Saved comparative analysis to {comparison_path} and {histogram_path}")

##############################################################################
# MAIN FUNCTION
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Causality Experiment - Process raw videos and compute causality scores")
    parser.add_argument("--model_interface", type=str, default="segformer",
                      choices=["segformer"], help="Model interface to use")
    parser.add_argument("--videos_dir", type=str, required=True,
                      help="Directory containing raw .mp4 video files")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for results and processed data")
    parser.add_argument("--n_blobs", type=int, default=2,
                      help="Number of blobs to detect and track (default: 2)")
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model interface
    if args.model_interface == "segformer":
        model_interface = SegFormerInterface()
    else:
        raise ValueError(f"Unknown model interface: {args.model_interface}")
    
    # Run experiment
    experiment = CausalityExperiment(model_interface, args.output_dir, args.n_blobs)
    
    try:
        experiment.run_full_experiment(
            videos_dir=args.videos_dir,
            resume=resume
        )
        print("Causality experiment completed successfully!")
        
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