import argparse
import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms as T
from PIL import Image
import numpy as np
from matplotlib import cm

torch.set_grad_enabled(False)

# Standard 91-class COCO labels

COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def main(args):
    # ------------------------------------------
    # 1) Load model (ignoring post-processor)
    # ------------------------------------------
    model, _ = torch.hub.load(
        'facebookresearch/detr',
        'detr_resnet101_panoptic',
        pretrained=False,
        return_postprocessor=True,
        num_classes=91
    )

    # ------------------------------------------
    # 2) Load custom checkpoint
    # ------------------------------------------
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        if "detr." in k:
            state_dict[k.replace("detr.", "")] = v
        else:
            state_dict[k] = v
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # ------------------------------------------
    # 3) Load the original image
    # ------------------------------------------
    original_img = Image.open(args.image_path).convert("RGB")
    orig_w, orig_h = original_img.size

    # We'll create a separate "resized" version of the image for the model
    # because DETR typically expects a certain size (we do short side=800).
    # Then we'll have to upsample the masks/bboxes back to the original size.
    resize_transform = T.Resize(800)  # short side to 800, aspect ratio preserved
    resized_img = resize_transform(original_img)
    resized_w, resized_h = resized_img.size

    # ------------------------------------------
    # 4) Full transform to feed to the model
    # ------------------------------------------
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Final model input
    model_in = transform(resized_img).unsqueeze(0)  # shape: (1, 3, resized_h, resized_w)

    # ------------------------------------------
    # 5) Inference
    # ------------------------------------------
    with torch.no_grad():
        outputs = model(model_in)
    # outputs["pred_logits"]: (1, 100, 92)
    # outputs["pred_boxes"] : (1, 100, 4)  in [0..1] (center_x, center_y, w, h)
    # outputs["pred_masks"] : (1, 100, mask_h, mask_w) (downsampled)

    pred_logits = outputs["pred_logits"]
    pred_boxes  = outputs["pred_boxes"]
    pred_masks  = outputs["pred_masks"]  # shape ~ (1, 100, resized_h/4, resized_w/4), typically

    # ------------------------------------------
    # 6) Get top-k predictions, ignoring "no object"
    # ------------------------------------------
    scores = pred_logits.softmax(-1)[..., :-1].max(-1)[0]  # shape: (1, 100)
    scores_squeezed = scores.squeeze(0)                    # shape: (100,)

    topk = args.top_k
    topk_scores, topk_indices = torch.topk(scores_squeezed, k=topk)

    # ------------------------------------------
    # 7) Prepare figure exactly in original size
    #    so the final saved image matches original pixels
    # ------------------------------------------
    dpi = 100.0  # dots per inch
    fig_w = orig_w / dpi
    fig_h = orig_h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    # Show the original image
    im_np = np.array(original_img)
    ax.imshow(im_np)
    ax.axis('off')

    # We'll threshold the masks at 0.5
    # and apply partial transparency (alpha)
    mask_threshold = 0.5
    alpha_val = 0.3  # high transparency so the background is clearly visible

    # Some distinct colormaps
    color_maps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']

    for i in range(topk):
        idx = topk_indices[i].item()
        score_val = topk_scores[i].item()

        # ---- Class label
        class_probs = pred_logits[0, idx, :-1].softmax(-1)
        class_id = class_probs.argmax().item()
        if class_id < len(COCO_CLASSES):
            label_str = COCO_CLASSES[class_id]
        else:
            label_str = f"Class {class_id}"

        # ---- 7a) BBOX in resized coords => scale to original coords
        # DETR bounding boxes are in [cx, cy, w, h] in relative coords, wrt the *resized* image
        cx, cy, w_box, h_box = pred_boxes[0, idx].tolist()
        # Convert relative coords to actual pixels in the resized image
        rx0 = (cx - 0.5 * w_box) * resized_w
        ry0 = (cy - 0.5 * h_box) * resized_h
        rx1 = (cx + 0.5 * w_box) * resized_w
        ry1 = (cy + 0.5 * h_box) * resized_h

        # Now scale from resized image to original
        scale_x = orig_w / float(resized_w)
        scale_y = orig_h / float(resized_h)
        x0 = rx0 * scale_x
        y0 = ry0 * scale_y
        x1 = rx1 * scale_x
        y1 = ry1 * scale_y

        # ---- 7b) Convert mask logits => upsample => threshold
        # pred_masks[0, idx] shape ~ (mask_h, mask_w) with mask_h,resized_h ~ resized_h/4
        mask_logit = pred_masks[0, idx].unsqueeze(0).unsqueeze(0)  # shape: (1,1,mask_h,mask_w)
        # Interpolate up to the original size
        upsampled_mask = F.interpolate(
            mask_logit,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        mask_prob = upsampled_mask.squeeze(0).squeeze(0).sigmoid().cpu().numpy()
        # Threshold
        binary_mask = mask_prob > mask_threshold

        # ---- 7c) Build RGBA overlay for the masked region
        cmap_name = color_maps[i % len(color_maps)]
        cmap_func = cm.get_cmap(cmap_name)
        color = cmap_func(0.6)  # pick the same spot in each colormap
        overlay_rgba = np.zeros((orig_h, orig_w, 4), dtype=np.float32)
        overlay_rgba[binary_mask, 0] = color[0]
        overlay_rgba[binary_mask, 1] = color[1]
        overlay_rgba[binary_mask, 2] = color[2]
        overlay_rgba[binary_mask, 3] = alpha_val  # partial transparency

        # ---- 7d) Show this overlay
        ax.imshow(overlay_rgba, interpolation='nearest')

        # ---- 7e) Place label near top-left of the box
        # Smaller font size => 12
        ax.text(
            x=x0,
            y=max(y0 - 5, 0),
            s=f"{label_str} ({score_val:.2f})",
            color="green",
            fontsize=12,
            fontweight='bold',
            zorder=999,
            bbox=dict(facecolor='black', alpha=0.5, pad=3)
        )

    plt.tight_layout()
    # Save with no extra padding so final image is exactly original size in pixels
    fig.savefig(args.output_path, bbox_inches='tight', pad_inches=0)
    print(f"Overlay image with top-{topk} predictions saved to {args.output_path}, size={orig_w}x{orig_h}px")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETR top-k instance overlay with original image size and no background spillover.")
    parser.add_argument("--model_path", required=True, help="Path to the custom DETR checkpoint (.pth file)")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--output_path", required=True, help="Path to save the resulting figure")
    parser.add_argument("--top_k", type=int, default=2, help="Number of top predictions to overlay (default=2)")
    args = parser.parse_args()
    main(args)
