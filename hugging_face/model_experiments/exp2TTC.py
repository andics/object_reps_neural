#!/usr/bin/env python3
"""
exp2TTC_new.py

Time-to-Collision (TTC) Experiment that integrates collision detection and participant correlation analysis.
Uses a configurable model interface for object detection and mask generation.

Usage:
    python exp2TTC_new.py --model_interface segformer --zip_path /path/to/videos.zip --csv_path /path/to/participants.csv --output_dir /path/to/output
"""

import argparse
import os
import sys
import json
import zipfile
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image

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
    1. Processes video frames using a model interface to detect objects
    2. Computes collision times under varying IoU thresholds
    3. Correlates model predictions with participant response times
    4. Generates analysis plots and statistics
    """
    
    def __init__(self, model_interface: ModelInterface, output_dir: str, logger: logging.Logger = None):
        self.model_interface = model_interface
        self.output_dir = output_dir
        self.logger = logger or self._setup_logger()
        
        # Initialize video processor
        self.video_processor = VideoProcessor(model_interface, self.logger)
        
        # Create output subdirectories
        self.results_dir = os.path.join(output_dir, "results")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.logs_dir = os.path.join(output_dir, "logs")
        self.temp_dir = os.path.join(output_dir, "temp_extraction")
        self.processed_videos_dir = os.path.join(output_dir, "processed_videos")
        
        for dir_path in [self.results_dir, self.plots_dir, self.logs_dir, self.temp_dir, self.processed_videos_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
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

    def run_full_experiment(self, zip_path: str, name_mapping_path: str, csv_path: str, 
                          iou_start: float = 0.05, iou_end: float = 0.95, iou_step: float = 0.05,
                          resume: bool = True) -> None:
        """Run the complete TTC experiment."""
        self.logger.info("Starting full TTC experiment")
        
        # Step 1: Extract video data
        self._extract_zip_if_needed(zip_path)
        
        # Step 2: Load mapping and participant data
        name_mapping = self._read_name_mapping(name_mapping_path)
        participant_df = self._read_participant_csv(csv_path)
        
        # Step 3: Generate collision detection data  
        iou_values = np.arange(iou_start, iou_end + iou_step, iou_step)
        iou_values = np.round(iou_values, decimals=3)
        
        collision_data = self._process_videos_for_collision_detection(iou_values, resume=resume)
        
        # Step 4: Analyze correlations with participant data
        if collision_data:
            self._analyze_participant_correlations(collision_data, name_mapping, participant_df, iou_values)
        else:
            self.logger.warning("No collision data generated - skipping correlation analysis")
        
        self.logger.info("TTC experiment completed successfully")

    def _extract_zip_if_needed(self, zip_path: str) -> None:
        """Extract zip file if not already extracted."""
        extract_dir = Path(self.temp_dir) / "videos_processed_copy"
        if not extract_dir.exists():
            self.logger.info(f"Extracting {zip_path} into {extract_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            self.logger.info("Extraction complete.")
        else:
            self.logger.info("Extraction directory already exists; skipping extraction.")

    def _read_name_mapping(self, name_mapping_path: str) -> Dict[str, str]:
        """Read the JSON name mapping."""
        self.logger.info(f"Reading name mapping from {name_mapping_path}...")
        with open(name_mapping_path, 'r') as f:
            mapping = json.load(f)
        self.logger.info(f"Loaded name mapping with {len(mapping)} entries.")
        return mapping

    def _read_participant_csv(self, csv_path: str) -> pd.DataFrame:
        """Read the participant CSV."""
        self.logger.info(f"Reading participant CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
        self.logger.info(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns.")
        return df

    def _process_videos_for_collision_detection(self, iou_values: np.ndarray, resume: bool = True) -> Dict[Tuple[str, float], float]:
        """Process all videos to detect collision times for different IoU thresholds."""
        self.logger.info("Starting collision detection across videos and IoU thresholds...")
        
        # Find video files in the extracted directory
        videos_dir = Path(self.temp_dir) / "videos_processed_copy"
        video_files = self._find_all_video_files(videos_dir)
        
        if not video_files:
            self.logger.error(f"No video files found in {videos_dir}")
            return {}
        
        collision_times = {}
        
        for video_file in video_files:
            video_name = self._get_video_name_from_path(video_file)
            self.logger.info(f"Processing video: {video_name}")
            
            # Use VideoProcessor to process the entire video
            try:
                video_output_dirs = self.video_processor.process_video(
                    video_path=str(video_file),
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

    def _generate_masks_for_video(self, subfolder: Path, frames_dir: Path) -> Dict[int, Dict[str, np.ndarray]]:
        """Generate object masks for all frames in a video using the model interface."""
        self.logger.info(f"Generating masks for video: {subfolder.name}")
        
        # Create output directory for masks
        masks_output_dir = subfolder / "frames_masks"
        masks_output_dir.mkdir(exist_ok=True)
        
        # Get all frame files
        frame_files = sorted([f for f in frames_dir.iterdir() 
                            if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        mask_data = {}
        
        for frame_file in frame_files:
            # Extract frame number
            frame_num = self._extract_frame_number(frame_file.name)
            
            # Load and process frame
            frame_image = Image.open(frame_file).convert('RGB')
            
            # Run inference
            predictions = self.model_interface.infer_image(frame_image)
            
            # Extract top 2 masks as blob_0 and blob_1
            pred_masks = predictions['pred_masks']  # Shape: (1, num_queries, H, W)
            
            frame_masks = {}
            for blob_idx in range(min(2, pred_masks.shape[1])):
                mask = pred_masks[0, blob_idx].cpu().numpy()  # (H, W)
                binary_mask = (mask > 0.5).astype(np.float32)
                
                # Save mask
                mask_filename = f"mask_memory_blob_{blob_idx}_frame_{frame_num:06d}.png"
                mask_path = masks_output_dir / mask_filename
                Image.fromarray((binary_mask * 255).astype(np.uint8)).save(mask_path)
                
                frame_masks[f"blob_{blob_idx}"] = binary_mask
            
            mask_data[frame_num] = frame_masks
        
        return mask_data

    def _find_all_video_files(self, videos_dir: Path) -> List[Path]:
        """Find all video files recursively in the directory."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(videos_dir.rglob(f"*{ext}"))
        
        return sorted(video_files)
    
    def _get_video_name_from_path(self, video_path: Path) -> str:
        """Extract video name from file path."""
        return video_path.stem.replace(" ", "+")
    
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
    
    def _extract_frame_number(self, filename: str) -> int:
        """Extract frame number from filename."""
        # Look for patterns like frame_000013.png or 000013.png
        import re
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0

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
                                        name_mapping: Dict[str, str], participant_df: pd.DataFrame, 
                                        iou_values: np.ndarray) -> None:
        """Analyze correlations between model predictions and participant data."""
        self.logger.info("Analyzing participant correlations...")
        
        # Group subfolders by model name
        model_to_subfolders = self._group_subfolders_by_model(collision_data)
        
        for model_name, subfolder_names in model_to_subfolders.items():
            for iou_thr in iou_values:
                self._analyze_model_iou_combination(
                    model_name, iou_thr, subfolder_names, collision_data, 
                    name_mapping, participant_df
                )

    def _group_subfolders_by_model(self, collision_data: Dict[Tuple[str, float], float]) -> Dict[str, List[str]]:
        """Group subfolders by model name."""
        model_to_subfolders = {}
        
        for (subfolder_name, _), _ in collision_data.items():
            model_name = self._parse_model_name_from_subfolder(subfolder_name)
            model_to_subfolders.setdefault(model_name, []).append(subfolder_name)
        
        # Remove duplicates
        for model_name in model_to_subfolders:
            model_to_subfolders[model_name] = list(set(model_to_subfolders[model_name]))
        
        return model_to_subfolders

    def _parse_model_name_from_subfolder(self, subfolder_name: str) -> str:
        """Extract model name from subfolder name."""
        # Example: "variable_pretrained_resnet101-BConcave+AConcave+3500" -> "variable_pretrained_resnet101"
        if '-' in subfolder_name:
            return subfolder_name.split('-', 1)[0]
        return "unknown_model"

    def _analyze_model_iou_combination(self, model_name: str, iou_thr: float, subfolder_names: List[str],
                                     collision_data: Dict[Tuple[str, float], float], name_mapping: Dict[str, str],
                                     participant_df: pd.DataFrame) -> None:
        """Analyze a specific model-IoU combination."""
        
        # Create output directory
        out_dir_name = f"{model_name}_IoU_{iou_thr}"
        out_dir = Path(self.results_dir) / out_dir_name
        out_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (out_dir / "ID").mkdir(exist_ok=True)
        (out_dir / "Average_person").mkdir(exist_ok=True)
        (out_dir / "concave_vs_convex").mkdir(exist_ok=True)
        
        # Individual participant analysis
        self._analyze_individual_participants(
            out_dir / "ID", iou_thr, subfolder_names, collision_data, name_mapping, participant_df
        )
        
        # Average participant analysis
        self._analyze_average_participant(
            out_dir / "Average_person", iou_thr, subfolder_names, collision_data, name_mapping, participant_df
        )
        
        # Concave vs convex analysis
        self._analyze_concave_vs_convex(
            out_dir / "concave_vs_convex", iou_thr, subfolder_names, collision_data, name_mapping, participant_df
        )

    def _analyze_individual_participants(self, output_dir: Path, iou_thr: float, subfolder_names: List[str],
                                       collision_data: Dict[Tuple[str, float], float], name_mapping: Dict[str, str],
                                       participant_df: pd.DataFrame) -> None:
        """Analyze individual participant correlations."""
        
        participant_groups = participant_df.groupby("ID")
        
        for pid, group in participant_groups:
            predicted_times = []
            human_times = []
            used_videos = []
            
            for _, row in group.iterrows():
                stimulus = row["stimulus"]
                subfolder = self._get_subfolder_for_stimulus(stimulus, name_mapping, subfolder_names)
                
                if subfolder:
                    collision_time = collision_data.get((subfolder, iou_thr), float('nan'))
                    if not np.isnan(collision_time):
                        predicted_times.append(collision_time)
                        human_times.append(row["rt"])
                        used_videos.append(subfolder)
            
            # Compute correlation
            r_val = self._compute_correlation(predicted_times, human_times) if len(predicted_times) > 1 else float('nan')
            
            # Create scatter plot
            fig, ax = plt.subplots()
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

    def _analyze_average_participant(self, output_dir: Path, iou_thr: float, subfolder_names: List[str],
                                   collision_data: Dict[Tuple[str, float], float], name_mapping: Dict[str, str],
                                   participant_df: pd.DataFrame) -> None:
        """Analyze average participant correlations."""
        
        # Map subfolders to average RTs
        subfolder_rts = {name: [] for name in subfolder_names}
        
        for _, row in participant_df.iterrows():
            stimulus = row["stimulus"]
            subfolder = self._get_subfolder_for_stimulus(stimulus, name_mapping, subfolder_names)
            if subfolder and subfolder in subfolder_rts:
                subfolder_rts[subfolder].append(row["rt"])
        
        # Compute averages and correlations
        avg_human_times = []
        model_times = []
        used_subfolders = []
        
        for subfolder_name in subfolder_names:
            rts = subfolder_rts[subfolder_name]
            if len(rts) > 0:
                avg_rt = np.mean(rts)
                collision_time = collision_data.get((subfolder_name, iou_thr), float('nan'))
                if not np.isnan(collision_time):
                    avg_human_times.append(avg_rt)
                    model_times.append(collision_time)
                    used_subfolders.append(subfolder_name)
        
        # Compute correlation
        r_val = self._compute_correlation(avg_human_times, model_times) if len(avg_human_times) > 1 else float('nan')
        
        # Create plot
        fig, ax = plt.subplots()
        ax.scatter(avg_human_times, model_times, c='red', alpha=0.7)
        ax.set_xlabel("Average Human RT (ms)")
        ax.set_ylabel("Model Collision Time (ms)")
        ax.set_title(f"Average Person, IoU={iou_thr}, r={r_val:.3f}")
        
        plt.savefig(output_dir / "average_person.png")
        plt.close(fig)
        
        with open(output_dir / "average_person.json", 'w') as f:
            json.dump({
                "correlation": r_val,
                "videos_used": used_subfolders,
                "data_points": len(avg_human_times)
            }, f, indent=2)

    def _analyze_concave_vs_convex(self, output_dir: Path, iou_thr: float, subfolder_names: List[str],
                                 collision_data: Dict[Tuple[str, float], float], name_mapping: Dict[str, str],
                                 participant_df: pd.DataFrame) -> None:
        """Analyze concave vs convex differences."""
        
        # Parse subfolders to determine concave/convex and ground truth
        gt_to_concave_vals = {}
        gt_to_convex_vals = {}
        
        for subfolder_name in subfolder_names:
            parsed_info = self._parse_subfolder_name(subfolder_name)
            gt = parsed_info.get("ground_truth")
            tokens = parsed_info.get("tokens", [])
            
            if gt is not None and len(tokens) >= 2:
                shape_token = tokens[1] if len(tokens) >= 2 else tokens[0]
                collision_time = collision_data.get((subfolder_name, iou_thr), float('nan'))
                
                if not np.isnan(collision_time):
                    if self._is_concave_token(shape_token):
                        gt_to_concave_vals.setdefault(gt, []).append(collision_time)
                    else:
                        gt_to_convex_vals.setdefault(gt, []).append(collision_time)
        
        # Compute differences for model
        gt_sorted = sorted(set(list(gt_to_concave_vals.keys()) + list(gt_to_convex_vals.keys())))
        model_diffs = []
        
        for gt in gt_sorted:
            concave_vals = gt_to_concave_vals.get(gt, [])
            convex_vals = gt_to_convex_vals.get(gt, [])
            
            mean_concave = np.mean(concave_vals) if concave_vals else float('nan')
            mean_convex = np.mean(convex_vals) if convex_vals else float('nan')
            diff = abs(mean_concave - mean_convex) if not (np.isnan(mean_concave) or np.isnan(mean_convex)) else float('nan')
            model_diffs.append(diff)
        
        # Compute differences for human data
        human_diffs = []
        df_grouped = participant_df.groupby("groundTruth")
        
        for gt in gt_sorted:
            if gt in df_grouped.groups:
                subdf = df_grouped.get_group(gt)
                concave_df = subdf[subdf["is_concave"] == 1]
                convex_df = subdf[subdf["is_concave"] == 0]
                
                mean_concave = concave_df["rt"].mean() if len(concave_df) > 0 else float('nan')
                mean_convex = convex_df["rt"].mean() if len(convex_df) > 0 else float('nan')
                diff = abs(mean_concave - mean_convex) if not (np.isnan(mean_concave) or np.isnan(mean_convex)) else float('nan')
            else:
                diff = float('nan')
            human_diffs.append(diff)
        
        # Create comparison plot
        x_indices = np.arange(len(gt_sorted))
        width = 0.3
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x_indices - width/2, model_diffs, width, label='Model', alpha=0.7)
        ax.bar(x_indices + width/2, human_diffs, width, label='Human', alpha=0.7)
        
        ax.set_xticks(x_indices)
        ax.set_xticklabels([str(g) for g in gt_sorted])
        ax.set_ylabel("Concave vs Convex (Absolute Difference in ms)")
        ax.set_title(f"Concave vs Convex Differences, IoU={iou_thr}")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "concave_vs_convex.png")
        plt.close(fig)

    # Helper methods
    def _get_subfolder_for_stimulus(self, stimulus: str, name_mapping: Dict[str, str], subfolder_names: List[str]) -> str:
        """Map stimulus to subfolder name."""
        base_stim = os.path.basename(stimulus)
        stim_id = os.path.splitext(base_stim)[0]
        
        if stim_id not in name_mapping:
            return None
        
        raw_folder_name = name_mapping[stim_id]
        folder_key = os.path.splitext(raw_folder_name)[0]
        
        for subfolder_name in subfolder_names:
            if folder_key in subfolder_name:
                return subfolder_name
        return None

    def _parse_subfolder_name(self, subfolder_name: str) -> Dict[str, Any]:
        """Parse subfolder name to extract components."""
        parts = subfolder_name.split('-', 1)
        model_name = parts[0]
        remainder = parts[1] if len(parts) > 1 else ""
        
        tokens = remainder.split('+')
        last_token = tokens[-1].replace("_flipped", "")
        
        try:
            ground_truth = int(last_token)
        except ValueError:
            ground_truth = None
        
        tokens = tokens[:-1]
        
        return {
            "model_name": model_name,
            "tokens": tokens,
            "ground_truth": ground_truth
        }

    def _is_concave_token(self, token: str) -> bool:
        """Check if token represents concave shape."""
        return "Concave" in token

    def _compute_correlation(self, xvals: List[float], yvals: List[float]) -> float:
        """Compute Pearson correlation."""
        if len(xvals) < 2:
            return float('nan')
        return np.corrcoef(xvals, yvals)[0, 1]


##############################################################################
# MAIN FUNCTION
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Run TTC Experiment with configurable model interface")
    
    parser.add_argument("--model_interface", default="segformer", 
                       choices=["segformer"], 
                       help="Model interface to use")
    parser.add_argument("--zip_path", required=True,
                       help="Path to the input .zip file containing the extracted frames")
    parser.add_argument("--name_mapping", required=True,
                       help="Path to the name_mapping.json file")
    parser.add_argument("--csv_path", required=True,
                       help="Path to the CSV file containing participant data")
    parser.add_argument("--output_dir", required=True,
                       help="Directory for experiment outputs")
    parser.add_argument("--iou_start", type=float, default=0.05,
                       help="Starting IoU threshold")
    parser.add_argument("--iou_end", type=float, default=0.95,
                       help="Ending IoU threshold")
    parser.add_argument("--iou_step", type=float, default=0.05,
                       help="IoU increment")
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
    experiment = TTCExperiment(model_interface, args.output_dir)
    
    # Determine resume flag
    resume = args.resume and not args.no_resume
    
    experiment.run_full_experiment(
        args.zip_path, args.name_mapping, args.csv_path,
        args.iou_start, args.iou_end, args.iou_step, resume=resume
    )


if __name__ == "__main__":
    main() 