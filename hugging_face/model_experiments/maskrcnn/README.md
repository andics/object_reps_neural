# Mask R-CNN Interface

This directory contains the Mask R-CNN model interface for use with the vision experiments. The interface provides DETR-compatible output format from Mask R-CNN models for seamless integration with existing experiment pipelines.

## Features

- **Instance Segmentation**: Uses Mask R-CNN's native instance segmentation capabilities
- **DETR-Compatible Output**: Converts Mask R-CNN outputs to match DETR format for experiment compatibility
- **Pre-trained Models**: Uses torchvision's pre-trained Mask R-CNN models
- **ResNet-50 Backbone**: Default model uses ResNet-50 with Feature Pyramid Network (FPN)
- **Real Masks**: Unlike DETR which generates masks from bounding boxes, Mask R-CNN provides actual segmentation masks

## Usage

### Basic Usage

```python
from maskrcnn.maskrcnn_interface import MaskRCNNInterface
from PIL import Image

# Initialize interface
interface = MaskRCNNInterface(model_name="maskrcnn_resnet50_fpn")
interface.load_model()

# Run inference
image = Image.open("your_image.jpg")
predictions = interface.infer_image(image)

print(f"Prediction format:")
print(f"- pred_masks shape: {predictions['pred_masks'].shape}")
print(f"- pred_logits shape: {predictions['pred_logits'].shape}")
print(f"- pred_boxes shape: {predictions['pred_boxes'].shape}")
```

### Using with Experiments

The Mask R-CNN interface can be used with any of the three experiments:

#### Experiment 1: Causality Analysis
```bash
python exp1Causality.py --model_interface maskrcnn --videos_dir /path/to/videos --output_dir /path/to/output
```

#### Experiment 2: Time-to-Collision (TTC)
```bash
python exp2TTC.py --model_interface maskrcnn --videos_dir /path/to/videos --csv_path /path/to/data.csv --output_dir /path/to/output
```

#### Experiment 3: Change Detection
```bash
python exp3Change.py --model_interface maskrcnn --images_dir /path/to/images --output_dir /path/to/output
```

### Custom Model Names

Currently supported models:
- `maskrcnn_resnet50_fpn` (default) - Mask R-CNN with ResNet-50 backbone and FPN

## Output Format

The interface returns predictions in DETR-compatible format:

- **`pred_masks`**: `torch.Tensor` of shape `(1, N, H, W)` where N is number of queries (default: 100)
  - Contains binary segmentation masks from Mask R-CNN
  - Actual instance segmentation masks (not converted from bounding boxes)
  
- **`pred_logits`**: `torch.Tensor` of shape `(1, N, num_classes)` 
  - Classification logits converted from Mask R-CNN scores
  - Uses COCO class labels (91 classes)
  
- **`pred_boxes`**: `torch.Tensor` of shape `(1, N, 4)` in DETR format
  - Bounding boxes in normalized (center_x, center_y, width, height) format

## Key Differences from SegFormer and DETR

| Feature | SegFormer | DETR | Mask R-CNN |
|---------|-----------|------|------------|
| Task | Semantic Segmentation | Object Detection | Instance Segmentation |
| Masks | Class-based regions | Boxes→Masks | True instance masks |
| Training Data | ADE20K | COCO | COCO |
| Backbone | MIT-B1 | ResNet-50 | ResNet-50+FPN |
| Output | Semantic labels | Detection boxes | Instance masks + boxes |

## Model Architecture

- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Training Dataset**: COCO (Common Objects in Context)
- **Classes**: 91 COCO object classes
- **Task**: Instance segmentation (detection + segmentation)

## Configuration Options

```python
interface = MaskRCNNInterface(
    model_name="maskrcnn_resnet50_fpn",  # Model architecture
    device="cuda",                       # Device for inference
    num_queries=100,                     # Number of output slots
    confidence_threshold=0.5,            # Detection confidence threshold
    pretrained=True                      # Use pre-trained weights
)
```

## Advantages for Vision Experiments

1. **True Instance Segmentation**: Provides actual pixel-level instance masks rather than converted bounding boxes
2. **High-Quality Masks**: Superior mask quality compared to box-based approaches
3. **Pre-trained Performance**: Excellent out-of-the-box performance on COCO objects
4. **Established Architecture**: Well-tested and widely-used model architecture

## Dependencies

```bash
pip install torch torchvision transformers pillow matplotlib numpy
```

## Example Output

When running the demo script:

```bash
python maskrcnn_interface.py
```

Output:
```
Prediction format:
- pred_masks shape: torch.Size([1, 100, 480, 640])
- pred_logits shape: torch.Size([1, 100, 91])
- pred_boxes shape: torch.Size([1, 100, 4])
Saved visualization to maskrcnn_demo_output.png
```

The interface automatically handles:
- Model loading and initialization
- Image preprocessing 
- Confidence filtering
- Format conversion to DETR-compatible output
- Mask and bounding box processing

This allows seamless integration with existing experiment pipelines while leveraging Mask R-CNN's superior instance segmentation capabilities. 