# Hugging Face Model Experiments

This directory contains self-contained experimental pipelines for analyzing object detection and segmentation using different model interfaces. Each experiment processes raw input data (videos or images) from scratch to final analysis.

## Overview

The experimental framework consists of:

- **Model Interfaces**: Configurable interfaces for different vision models (currently SegFormer)
- **VideoProcessor**: Comprehensive video processing pipeline with blob detection and tracking
- **Three Main Experiments**: Complete pipelines from raw data to analysis results

## Experiments

### Experiment 1: Causality Analysis (`exp1Causality.py`)

**Purpose**: Analyze causal relationships between objects based on distance changes over time.

**Input**: Directory of raw `.mp4` video files

**Process**:
1. **Video Processing**: Extract frames from each video using VideoProcessor
2. **Object Detection**: Detect and track objects (blobs) across frames
3. **Distance Computation**: Calculate distance changes between object centroids
4. **Causality Scoring**: Compute causality scores based on distance change patterns
5. **Analysis**: Generate plots and statistical analysis of causality scores

**Key Features**:
- Memory-based object tracking with exponential decay (alpha=0.7)
- Skip initial frames (13 frames) for stabilization
- Distance change rate analysis for causality scoring
- Comparative analysis across all input videos

**Output Structure**:
```
output_dir/
├── processed_videos/          # VideoProcessor output for each video
├── distance_data/            # CSV files with distance measurements
├── results/                  # JSON and CSV summary files
├── plots/                   # Individual and comparative plots
└── logs/                    # Processing logs
```

**Usage**:
```bash
python exp1Causality.py --model_interface segformer \
                        --videos_dir /path/to/raw_videos \
                        --output_dir /path/to/output \
                        [--n_blobs 2] \
                        [--resume]
```

### Experiment 2: Time-to-Collision (TTC) Analysis (`exp2TTC.py`)

**Purpose**: Detect collision times at various IoU thresholds and correlate with human response data.

**Input**: 
- Directory of raw `.mp4` video files
- CSV file with participant response data

**Process**:
1. **Video Processing**: Extract frames and detect objects using VideoProcessor
2. **Collision Detection**: Find first collision time for each IoU threshold (0.05 to 0.95)
3. **Correlation Analysis**: Compare model collision times with participant response times
4. **Statistical Analysis**: Individual participant and average correlations

**Key Features**:
- IoU-based collision detection across multiple thresholds
- Participant correlation analysis (individual and average)
- Resume functionality for long processing runs
- Comprehensive statistical summaries

**Output Structure**:
```
output_dir/
├── processed_videos/          # VideoProcessor output for each video
├── results/                  # Results organized by IoU threshold
│   └── IoU_0.05/
│       ├── ID/              # Individual participant analyses
│       ├── Average_person/  # Average participant analysis
│       └── summary/         # Summary statistics
├── plots/                   # Correlation plots and summaries
└── logs/                    # Processing logs
```

**Usage**:
```bash
python exp2TTC_b1.py --model_interface segformer \
                  --videos_dir /path/to/raw_videos \
                  --csv_path /path/to/participant_data.csv \
                  --output_dir /path/to/output \
                  [--n_blobs 2] \
                  [--iou_start 0.05] [--iou_end 0.95] [--iou_step 0.05] \
                  [--resume]
```

### Experiment 3: Change Detection (`exp3Change.py`)

**Purpose**: Segment blobs in individual images and analyze change detection across different thresholds.

**Input**: Directory of raw image files (PNG, JPG, etc.)

**Process**:
1. **Image Processing**: Process each image individually for blob detection
2. **Blob Segmentation**: Use model interface to segment detected blobs
3. **Threshold Analysis**: Analyze segmentation quality across different thresholds
4. **Mistake Scoring**: Compute mistake scores based on threshold comparisons

**Key Features**:
- Individual image processing (no temporal dependencies)
- Multi-threshold analysis for segmentation quality
- Blob detection using intensity thresholding
- Comprehensive mistake score analysis

**Output Structure**:
```
output_dir/
├── processed_images/         # Processed output for each image
│   └── segformer_model_image1/
│       ├── frames_blobs/    # Blob visualizations
│       ├── frames_masks_nonmem/  # Binary masks
│       ├── frames_collage/  # Top candidate comparisons
│       └── frames_processed/ # Final overlays
├── threshold_results/       # Results for each threshold
│   └── 1_comparison/
│       ├── results.json
│       ├── mistake_scores.png
│       └── mistake_distribution.png
├── results/                 # Summary analyses
├── plots/                   # Comparative threshold plots
└── logs/                    # Processing logs
```

**Usage**:
```bash
python exp3Change.py --model_interface segformer \
                     --images_dir /path/to/raw_images \
                     --output_dir /path/to/output \
                     [--thresholds 1 2 3 4 5 6 8 10 12 14 16 18 20] \
                     [--resume]
```

## VideoProcessor Class

The `VideoProcessor` class (`video_processor.py`) provides the core video processing functionality used by experiments 1 and 2.

### Key Features

**Video Processing Pipeline**:
- Frame extraction using imageio
- Blob detection with intensity thresholding (black_thresh=30)
- Model inference using configurable interfaces
- Bipartite assignment for mask-to-blob matching using IoU optimization

**Memory-Based Tracking**:
- Exponential decay memory system (alpha=0.7)
- Cross-frame object consistency
- Temporal smoothing of object masks

**Output Generation**:
- Multiple mask formats (memory and non-memory)
- Visualization overlays with polygons
- Final processed video creation
- Organized directory structure

**Resume Functionality**:
- Automatic checkpoint detection
- Resume from last processed frame
- Metadata preservation

**Configurable Blob Detection**:
- Number of blobs to detect and track can be specified via `n_blobs` parameter
- Default value is 2 blobs (suitable for most collision/interaction scenarios)
- Supports any number of blobs based on experimental requirements
- Memory arrays and tracking automatically adjust to specified blob count

### VideoProcessor Usage

```python
from video_processor import VideoProcessor
from segformer.segformer_interface import SegFormerInterface

# Initialize with custom number of blobs
model_interface = SegFormerInterface()
processor = VideoProcessor(model_interface, n_blobs=3, logger)

# Process video
output_dirs = processor.process_video(
    video_path="/path/to/video.mp4",
    output_root="/path/to/output",
    model_prefix="segformer_model",
    resume=True
)
```

### Output Directory Structure

VideoProcessor creates a comprehensive directory structure:

```
output_root/
└── segformer_model-video_name/
    ├── frames_blobs/            # Blob detection visualizations
    ├── frames_masks/            # Memory-based masks
    ├── frames_masks_nonmem/     # Non-memory masks
    ├── frames_processed/        # Final overlay frames
    ├── frames_collage/          # Candidate mask comparisons
    ├── videos_processed/        # Final output video
    └── metadata/               # Processing metadata and status
```

## Model Interfaces

### SegFormerInterface

Located in `segformer/segformer_interface.py`, this interface provides:

- **DETR-compatible Output**: Standardized prediction format
- **Automatic Model Loading**: Lazy loading with device management
- **Batch Processing**: Efficient inference on image batches
- **Memory Management**: Automatic GPU/CPU handling

**Interface Contract**:
```python
def infer_image(self, image: PIL.Image) -> Dict[str, torch.Tensor]:
    # Returns: {'pred_masks': tensor of shape (1, num_queries, H, W)}
```

## Installation and Setup

### Requirements

- Python 3.8+
- PyTorch with CUDA support (recommended)
- Transformers library
- Additional dependencies: see requirements.txt

### Model Setup

1. **SegFormer Model**: Automatically downloaded via Hugging Face transformers
2. **Custom Models**: Place checkpoint files in appropriate directories
3. **GPU Support**: CUDA-enabled PyTorch recommended for faster processing

## Common Usage Patterns

### Basic Experiment Run

```bash
# Experiment 1: Causality Analysis
python exp1Causality.py --model_interface segformer \
                        --videos_dir /data/raw_videos \
                        --output_dir /output/exp1_results \
                        --n_blobs 2

# Experiment 2: TTC Analysis  
python exp2TTC_b1.py --model_interface segformer \
                  --videos_dir /data/raw_videos \
                  --csv_path /data/participants.csv \
                  --output_dir /output/exp2_results \
                  --n_blobs 2

# Experiment 3: Change Detection
python exp3Change.py --model_interface segformer \
                     --images_dir /data/raw_images \
                     --output_dir /output/exp3_results
```

### Resume from Interruption

All experiments support automatic resume functionality:

```bash
# Resume automatically (default)
python exp1Causality.py --resume [other args...]

# Force restart from scratch
python exp1Causality.py --no_resume [other args...]
```

### Batch Processing

For large datasets, experiments can be run in sequence or parallel:

```bash
# Sequential processing
for exp in exp1Causality.py exp2TTC_b1.py exp3Change.py; do
    python $exp --model_interface segformer [args...]
done

# Parallel processing (if sufficient resources)
python exp1Causality.py [args...] &
python exp2TTC_b1.py [args...] &
python exp3Change.py [args...] &
wait
```

### Configuring Blob Detection

The number of blobs to detect can be customized based on experimental scenarios:

```bash
# Two-object interaction (default)
python exp1Causality.py --n_blobs 2 [other args...]

# Multi-object scenarios (3+ objects)
python exp1Causality.py --n_blobs 4 [other args...]

# Single object tracking
python exp2TTC_b1.py --n_blobs 1 [other args...]

# Complex scenes with many objects
python exp1Causality.py --n_blobs 6 [other args...]
```

Note: The number of blobs detected affects:
- Memory requirements (scales linearly with blob count)
- Processing time (slightly increases with more blobs)
- Analysis complexity (distance calculations scale with blob pairs)

## Performance Considerations

### Memory Usage

- **GPU Memory**: ~4-8GB for SegFormer inference
- **RAM**: ~2-4GB per video being processed
- **Storage**: Processed videos require ~5-10x original size

### Processing Time

- **Video Processing**: ~1-2 minutes per minute of video (GPU)
- **Image Processing**: ~1-5 seconds per image (GPU)
- **Analysis**: Generally < 1 minute for typical datasets

### Optimization Tips

1. **Use GPU**: Significantly faster than CPU-only processing
2. **Enable Resume**: Prevents loss of progress on interruptions
3. **Batch Size**: Adjust based on available GPU memory
4. **Storage**: Use SSD for faster I/O during processing

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller models
2. **Missing Dependencies**: Check requirements.txt installation
3. **File Path Issues**: Use absolute paths when possible
4. **Model Loading**: Ensure internet connection for Hugging Face models

### Debug Logging

All experiments provide detailed logging:

```bash
# Check log files in output_dir/logs/
tail -f /path/to/output/logs/experiment_TIMESTAMP.log
```

### Resume Issues

If resume functionality fails:

```bash
# Force clean restart
python experiment.py --no_resume [other args...]
```

## Input Data Requirements

### Video Files (Experiments 1 & 2)

- **Format**: .mp4 (recommended), other formats supported by imageio
- **Resolution**: Any resolution, automatically handled
- **Frame Rate**: Preserved in output, 30fps default for metadata
- **Content**: Videos should contain detectable objects/blobs

### Image Files (Experiment 3)

- **Formats**: PNG, JPG, JPEG, BMP, TIF, TIFF
- **Resolution**: Any resolution, automatically handled  
- **Content**: Images should contain detectable blobs/objects

### Participant Data (Experiment 2)

CSV file with columns:
- `ID`: Participant identifier
- `stimulus`: Stimulus identifier (for matching with videos)
- `rt`: Response time in milliseconds
- Additional columns preserved but not used

## Expected Output Formats

### Analysis Results

- **JSON**: Structured results with metadata
- **CSV**: Tabular data for further analysis
- **PNG**: High-resolution plots and visualizations

### Processed Data

- **Videos**: Processed videos with object overlays
- **Masks**: Binary masks in PNG format
- **Visualizations**: Debug and analysis plots

### Summary Statistics

- **Correlations**: Participant correlation analyses
- **Distributions**: Score and timing distributions  
- **Comparisons**: Threshold and model comparisons

This comprehensive framework provides a complete solution for object detection and analysis experiments, from raw data input to final statistical analysis and visualization.