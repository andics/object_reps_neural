 # Model Experiments Framework

This directory contains a refactored experiment framework that uses configurable model interfaces for object detection and segmentation tasks. The framework supports multiple experiments with a consistent interface pattern and includes comprehensive video processing capabilities.

## Overview

The framework consists of:
1. **Model Interface Classes** - Standardized interfaces for different models
2. **Video Processor** - Complete video processing pipeline with frame extraction, model inference, and mask generation
3. **Consolidated Experiment Files** - Self-contained experiments that handle everything from raw videos/images to final analysis
4. **Rich Logging and Output Structure** - Comprehensive logging and organized output directories
5. **Checkpoint/Resume System** - Intelligent resumption of processing to avoid starting from zero

## Model Interfaces

### SegFormerInterface

Located in `segformer/segformer_interface.py`, this interface provides:

- **DETR-compatible output format** for experiments originally designed for DETR models
- **Semantic-to-instance mask conversion** for compatibility with object detection workflows
- **Configurable model loading** from HuggingFace Hub
- **Standard inference API** that all experiments can use

**Key Methods:**
- `load_model()` - Load the SegFormer model
- `infer_image(image)` - Run inference and return DETR-format predictions
- Returns: `{'pred_masks': tensor, 'pred_logits': tensor, 'pred_boxes': tensor}`

## Video Processor

Located in `video_processor.py`, this class provides comprehensive video processing capabilities:

**Key Features:**
- **Complete Video Pipeline** - Handles video reading, frame extraction, model inference, and output generation
- **Blob Detection and Tracking** - Detects colored blobs and tracks them across frames using memory-based masks
- **Bipartite Assignment** - Intelligently matches predicted masks to detected blobs using IoU optimization
- **Checkpoint/Resume** - Saves processing state and can resume from interruption
- **Organized Output Structure** - Creates comprehensive directory structure with masks, visualizations, and metadata

**Key Methods:**
- `process_video(video_path, output_root, model_prefix, resume=True)` - Process complete video through pipeline
- `setup_output_directories()` - Create organized output directory structure
- `check_processing_status()` - Determine what processing has been completed

**Output Structure:**
```
{model_prefix}-{video_name}/
├── frames_blobs/          # Blob detection visualizations
├── frames_masks/          # Memory-based masks
├── frames_masks_nonmem/   # Immediate assignment masks  
├── frames_processed/      # Final overlay frames
├── videos_processed/      # Final processed video
└── metadata/             # Processing status and metadata
```

## Experiments

### Experiment 1: Causality Analysis (`exp1Causality.py`)

**Self-contained workflow** that processes video files, computes collision distances between objects, and generates causality plots.

**Process:**
1. Processes .mp4 video files using VideoProcessor
2. Extracts frames and generates object masks
3. Computes collision distances for multiple IoU thresholds  
4. Generates causality correlation plots and statistical analysis

**Usage:**
```bash
python exp1Causality.py \
    --model_interface segformer \
    --data_dir /path/to/videos \
    --output_dir /path/to/output \
    --resume  # Resume from previous processing (default)
```

**Outputs:**
- `processed_videos/` - Complete video processing outputs for each input video
- `results/collision_distances_*.csv` - Distance measurements for different thresholds
- `plots/causality_plot_*.png` - Causality correlation plots
- `plots/boundary_detailed.json` - Detailed boundary analysis results
- `plots/centroid_detailed.json` - Detailed centroid analysis results
- `logs/causality_exp_*.log` - Detailed execution logs

### Experiment 2: Time-to-Collision (`exp2TTC.py`)

**Self-contained workflow** that processes videos, analyzes collision detection timing, and correlates with participant response data.

**Process:**
1. Extracts video files from ZIP archive
2. Processes each video using VideoProcessor to generate masks
3. Computes collision times across multiple IoU thresholds
4. Correlates model predictions with human participant data
5. Generates correlation plots and statistical analysis

**Usage:**
```bash
python exp2TTC.py \
    --model_interface segformer \
    --zip_path /path/to/videos.zip \
    --name_mapping /path/to/mapping.json \
    --csv_path /path/to/participants.csv \
    --output_dir /path/to/output \
    --resume  # Resume from previous processing (default)
```

**Outputs:**
- `processed_videos/` - Complete video processing outputs for each input video
- `results/{model}_IoU_{threshold}/ID/` - Individual participant correlations
- `results/{model}_IoU_{threshold}/Average_person/` - Average participant analysis
- `results/{model}_IoU_{threshold}/concave_vs_convex/` - Shape comparison analysis
- `logs/ttc_exp_*.log` - Detailed execution logs

### Experiment 3: Change Detection (`exp3Change.py`)

**Self-contained workflow** that processes individual images, segments blobs, and analyzes change detection across different thresholds.

**Process:**
1. Processes individual image files using model interface
2. Detects blobs using intensity thresholding
3. Generates candidate masks using model inference
4. Selects best masks based on IoU with detected blobs
5. Analyzes before/after image pairs for area changes
6. Computes detection rates across multiple thresholds

**Usage:**
```bash
python exp3Change.py \
    --model_interface segformer \
    --images_folder /path/to/images \
    --output_dir /path/to/output \
    --resume  # Resume from previous processing (default)
```

**Outputs:**
- `processed_images/` - Segmented images with masks and visualizations
- `threshold_results/{threshold}_comparison/` - Analysis for each detection threshold
- `plots/` - Summary plots and visualizations
- `logs/change_exp_*.log` - Detailed execution logs

## Key Features

### 1. **Complete Self-Contained Workflow**
- Each experiment handles everything from raw videos/images to final analysis
- Integrated video processing using VideoProcessor class
- No need to run separate preprocessing scripts
- Automatic model loading and inference management

### 2. **Intelligent Checkpoint/Resume System**
- Automatically saves processing progress and metadata
- Can resume from any interruption point
- Avoids reprocessing already completed frames/videos
- Smart detection of existing outputs

### 3. **Model Interface Abstraction**
- Experiments are decoupled from specific model implementations
- Easy to add new model interfaces (YOLO, SAM, etc.)
- Consistent API across all experiments
- DETR-compatible output format

### 4. **Comprehensive Video Processing**
- Complete video pipeline from .mp4 files to analysis-ready data
- Blob detection and memory-based tracking across frames
- Bipartite mask assignment using IoU optimization
- Multiple output formats (masks, visualizations, processed videos)

### 5. **Rich Logging and Progress Tracking**
- Comprehensive logging at DEBUG and INFO levels
- Timestamped log files for reproducibility
- Real-time progress tracking with frame/video counts
- Detailed error reporting and recovery information

### 6. **Organized Output Structure**
- Consistent directory structure across experiments
- Separate subdirectories for different output types
- JSON metadata for programmatic access to results
- Automatic cleanup and organization

## Adding New Model Interfaces

To add a new model interface:

1. **Create interface class** inheriting from `ModelInterface`:
```python
class YourModelInterface(ModelInterface):
    def load_model(self):
        # Load your model
        pass
    
    def infer_image(self, image):
        # Return DETR-compatible format
        return {
            'pred_masks': masks_tensor,
            'pred_logits': logits_tensor, 
            'pred_boxes': boxes_tensor
        }
```

2. **Update experiment files** to support the new interface:
```python
if args.model_interface == "your_model":
    model_interface = YourModelInterface(model_name=args.model_name)
```

## Original vs. New Files

### Original Files (preserved):
- `exp1Causality_1_dist.py` - Distance computation only
- `exp1Causality_2_plots.py` - Plotting only  
- `exp2TTC.py` - Original TTC implementation
- `exp3Change_1_segment_blobs.py` - Blob segmentation only
- `exp3Change_2_extract_mistake_score.py` - Mistake scoring only

### New Consolidated Files:
- `exp1Causality.py` - Complete causality analysis workflow
- `exp2TTC_new.py` - Complete TTC analysis workflow
- `exp3Change.py` - Complete change detection workflow
- `segformer/segformer_interface.py` - Updated model interface

## Dependencies

```bash
pip install transformers safetensors huggingface_hub pillow matplotlib torch torchvision numpy pandas scipy scikit-image
```

## Notes

- The framework is designed to be immediately runnable out of the box
- All experiments include proper error handling and recovery
- Log files provide detailed information for debugging and reproducibility
- Output formats are designed to be compatible with existing analysis pipelines