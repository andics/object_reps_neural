 # Model Experiments Framework

This directory contains a refactored experiment framework that uses configurable model interfaces for object detection and segmentation tasks. The framework supports multiple experiments with a consistent interface pattern.

## Overview

The framework consists of:
1. **Model Interface Classes** - Standardized interfaces for different models
2. **Consolidated Experiment Files** - Complete experiments that integrate multiple processing steps
3. **Rich Logging and Output Structure** - Comprehensive logging and organized output directories

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

## Experiments

### Experiment 1: Causality Analysis (`exp1Causality.py`)

Computes collision distances between objects and generates causality plots.

**Usage:**
```bash
python exp1Causality.py \
    --model_interface segformer \
    --data_dir /path/to/video/frames \
    --output_dir /path/to/output
```

**Outputs:**
- `results/collision_distances_*.csv` - Distance measurements for different thresholds
- `plots/causality_plot_*.png` - Causality correlation plots
- `plots/boundary_detailed.json` - Detailed boundary analysis results
- `plots/centroid_detailed.json` - Detailed centroid analysis results
- `logs/causality_exp_*.log` - Detailed execution logs

### Experiment 2: Time-to-Collision (`exp2TTC_new.py`)

Analyzes collision detection timing and correlates with participant response data.

**Usage:**
```bash
python exp2TTC_new.py \
    --model_interface segformer \
    --zip_path /path/to/videos.zip \
    --name_mapping /path/to/mapping.json \
    --csv_path /path/to/participants.csv \
    --output_dir /path/to/output
```

**Outputs:**
- `results/{model}_IoU_{threshold}/ID/` - Individual participant correlations
- `results/{model}_IoU_{threshold}/Average_person/` - Average participant analysis
- `results/{model}_IoU_{threshold}/concave_vs_convex/` - Shape comparison analysis
- `logs/ttc_exp_*.log` - Detailed execution logs

### Experiment 3: Change Detection (`exp3Change.py`)

Segments blobs in images and analyzes change detection across different thresholds.

**Usage:**
```bash
python exp3Change.py \
    --model_interface segformer \
    --images_folder /path/to/images \
    --output_dir /path/to/output
```

**Outputs:**
- `processed_images/` - Segmented images with masks and visualizations
- `threshold_results/{threshold}_comparison/` - Analysis for each detection threshold
- `plots/` - Summary plots and visualizations
- `logs/change_exp_*.log` - Detailed execution logs

## Key Features

### 1. **Model Interface Abstraction**
- Experiments are decoupled from specific model implementations
- Easy to add new model interfaces (YOLO, SAM, etc.)
- Consistent API across all experiments

### 2. **Consolidated Processing**
- Each experiment integrates multiple processing steps
- No need to run separate scripts for data processing and analysis
- Streamlined workflows with automatic dependency handling

### 3. **Rich Logging**
- Comprehensive logging at DEBUG and INFO levels
- Timestamped log files for reproducibility
- Progress tracking and error reporting

### 4. **Organized Output Structure**
- Consistent directory structure across experiments
- Separate subdirectories for different output types
- JSON metadata for programmatic access to results

### 5. **DETR Compatibility**
- Original experiments designed for DETR continue to work
- SegFormer output converted to DETR-compatible format
- Maintains existing analysis pipelines

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