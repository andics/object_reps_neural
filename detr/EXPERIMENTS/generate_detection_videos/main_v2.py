import torch
import torchvision.transforms as T
from PIL import Image
from collections import OrderedDict
import argparse
import io

# For plotting/saving
import matplotlib.pyplot as plt
import numpy as np

# ...existing code for your 91-class COCO definition (if needed)...

def main(args):
    # Load model + postprocessor from hub, set pretrained=False, num_classes=91
    model, postprocessor = torch.hub.load(
        'facebookresearch/detr',
        'detr_resnet101_panoptic',
        pretrained=False,
        return_postprocessor=True,
        num_classes=91
    )
    # Load custom checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        if "detr." in k:
            k_n = k.replace("detr.", "")
            state_dict[k_n] = v
        else:
            state_dict[k] = v
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Define transform, as in the notebook
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load and transform the image
    im = Image.open(args.image_path).convert("RGB")
    img = transform(im).unsqueeze(0)

    # Forward pass
    out = model(img)

    # Postprocess to get panoptic segmentation
    # We provide the (height, width) as a tensor
    result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

    # Retrieve the special-format PNG from the result, then save it
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    panoptic_seg.save(args.output_path)

    print(f"Panoptic result saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR panoptic inference script (notebook-based)")
    parser.add_argument("--model_path", required=True, help="Path to the custom panoptic DETR checkpoint (.pth file)")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--output_path", required=True, help="Path to save the panoptic segmentation PNG")
    args = parser.parse_args()
    main(args)
