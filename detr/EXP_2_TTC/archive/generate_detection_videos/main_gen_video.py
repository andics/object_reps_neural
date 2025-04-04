#!/usr/bin/env python3
"""
main_gen_video.py

Reads an input .mp4 video using imageio, processes each frame with a DETR model
(overlays the top-k masks with partial transparency and bounding boxes),
and writes out a new .mp4 video (also via imageio).

No direct usage of OpenCV or external ffmpeg commands, so we avoid the
"cv2.dnn.DictValue" error and do not rely on 'ffmpeg' being in PATH.

Resumable: we optionally store intermediate frames to disk (like before) to
avoid re-running if interrupted. If you do NOT want disk usage, set
--resumable=False (in which case frames are processed in-memory).
"""

import argparse
import os
import sys
import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from collections import OrderedDict
from PIL import Image
import numpy as np

import imageio
# If you get errors about missing backend, you might try: pip install imageio-ffmpeg

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm

torch.set_grad_enabled(False)

# -----------------------------------------------------------------------------
# OPTIONAL MONKEY-PATCH FOR THE "cv2.dnn.DictValue" ERROR:
# If somewhere else in your environment forcibly tries to load cv2:
#
# import sys, types
# module_name = 'cv2.dnn'
# if module_name not in sys.modules:
#     dummy_dnn = types.ModuleType(module_name)
#     setattr(dummy_dnn, 'DictValue', type('DictValue', (object,), {}))
#     sys.modules[module_name] = dummy_dnn
#
# -----------------------------------------------------------------------------
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def load_model(model_path):
    """
    Load DETR (panoptic) from torch hub, then load custom checkpoint.
    """
    print("Loading model from torch hub (no postprocessor).")
    model, _ = torch.hub.load(
        'facebookresearch/detr',
        'detr_resnet101_panoptic',
        pretrained=False,
        return_postprocessor=True,
        num_classes=91
    )

    print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        if "detr." in k:
            state_dict[k.replace("detr.", "")] = v
        else:
            state_dict[k] = v

    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def detect_and_overlay(model, pil_img, top_k=2):
    """
    Perform inference on a single PIL image with the given DETR model,
    overlay the top-k masks with partial transparency, bounding boxes, etc.,
    returning the processed image as a NumPy array (RGB).
    """
    orig_w, orig_h = pil_img.size

    # Resize for DETR (short side = 800)
    resize_transform = T.Resize(800)  # short side to 800
    resized_img = resize_transform(pil_img)
    resized_w, resized_h = resized_img.size

    # Convert to tensor & normalize
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model_input = transform(resized_img).unsqueeze(0)  # (1,3,H,W)

    with torch.no_grad():
        outputs = model(model_input)

    pred_logits = outputs["pred_logits"]  # (1,100,92)
    pred_boxes  = outputs["pred_boxes"]   # (1,100,4)
    pred_masks  = outputs["pred_masks"]   # (1,100,mask_h,mask_w)

    # Confidence ignoring last class "no object"
    scores = pred_logits.softmax(-1)[..., :-1].max(-1)[0]  # (1,100)
    scores_squeezed = scores.squeeze(0)                    # (100,)
    topk_scores, topk_indices = torch.topk(scores_squeezed, k=top_k)

    # We'll threshold masks at 0.5 and use alpha=0.3
    mask_threshold = 0.5
    alpha_val = 0.3

    im_np = np.array(pil_img)

    dpi = 100.0
    fig_w = orig_w / dpi
    fig_h = orig_h / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(im_np)
    ax.axis('off')

    color_maps = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples']

    for i in range(top_k):
        idx = topk_indices[i].item()
        score_val = topk_scores[i].item()

        # Class label
        class_probs = pred_logits[0, idx, :-1].softmax(-1)
        class_id = class_probs.argmax().item()
        if class_id < len(COCO_CLASSES):
            label_str = COCO_CLASSES[class_id]
        else:
            label_str = f"Class {class_id}"

        # BBOX: Convert from relative coords in the resized image to original coords
        cx, cy, w_box, h_box = pred_boxes[0, idx].tolist()
        rx0 = (cx - 0.5 * w_box) * resized_w
        ry0 = (cy - 0.5 * h_box) * resized_h
        rx1 = (cx + 0.5 * w_box) * resized_w
        ry1 = (cy + 0.5 * h_box) * resized_h

        scale_x = orig_w / float(resized_w)
        scale_y = orig_h / float(resized_h)
        x0 = rx0 * scale_x
        y0 = ry0 * scale_y
        x1 = rx1 * scale_x
        y1 = ry1 * scale_y

        # Upsample mask to the original resolution
        mask_logit = pred_masks[0, idx].unsqueeze(0).unsqueeze(0)
        upsampled_mask = F.interpolate(
            mask_logit,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        mask_prob = upsampled_mask.squeeze(0).squeeze(0).sigmoid().cpu().numpy()
        binary_mask = mask_prob > mask_threshold

        # RGBA overlay
        cmap_name = color_maps[i % len(color_maps)]
        cmap_func = cm.get_cmap(cmap_name)
        color = cmap_func(0.6)
        overlay_rgba = np.zeros((orig_h, orig_w, 4), dtype=np.float32)
        overlay_rgba[binary_mask, 0] = color[0]
        overlay_rgba[binary_mask, 1] = color[1]
        overlay_rgba[binary_mask, 2] = color[2]
        overlay_rgba[binary_mask, 3] = alpha_val

        ax.imshow(overlay_rgba, interpolation='nearest')

        # Text label
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

    # Render to numpy array
    fig.canvas.draw()
    out_width, out_height = fig.canvas.get_width_height()
    out_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    out_img = out_img.reshape((out_height, out_width, 3))
    plt.close(fig)

    return out_img


def main():
    parser = argparse.ArgumentParser(description="Generate a .mp4 video with top-k DETR overlays on each frame (using imageio).")
    parser.add_argument("--model_path", required=True, help="Path to the custom DETR checkpoint (.pth file)")
    parser.add_argument("--video_path", required=True, help="Path to the input .mp4 video")
    parser.add_argument("--output_video_path", required=True, help="Path to the output .mp4 video")
    parser.add_argument("--top_k", type=int, default=2, help="Number of top predictions to overlay (default=2)")
    parser.add_argument("--resumable", action="store_true", default=True,
                        help="If true, store intermediate frames on disk so we can resume if interrupted.")
    args = parser.parse_args()

    # 1) Load the model
    model = load_model(args.model_path)

    # 2) We'll read the input video with imageio
    print(f"Reading input video {args.video_path}")
    video_reader = imageio.get_reader(args.video_path, "ffmpeg")  # backend might be 'pyav' or others
    meta_data = video_reader.get_meta_data()
    fps = meta_data.get('fps', 30)  # fallback to 30
    # Some backends might store 'size', but let's just read frames to find shape.

    # We create local storage folders if we're doing a resumable approach
    frames_raw_dir = "frames_raw"
    frames_proc_dir = "frames_processed"
    if args.resumable:
        if not os.path.exists(frames_raw_dir):
            os.makedirs(frames_raw_dir)
        if not os.path.exists(frames_proc_dir):
            os.makedirs(frames_proc_dir)

    # 3) Collect frames in memory or on disk
    # We'll also guess the video size from the first frame
    raw_frames_paths = []
    processed_frames_paths = []

    frame_idx = 0
    for frame in video_reader:
        # 'frame' is a numpy array of shape (height, width, channels)
        # Convert it to PIL
        if frame_idx == 0:
            video_h, video_w, _ = frame.shape
            print(f"Video resolution is {video_w}x{video_h}, fps={fps}")

        if args.resumable:
            # write raw frame to frames_raw_dir if not exist
            raw_path = os.path.join(frames_raw_dir, f"frame_{frame_idx:06d}.png")
            proc_path = os.path.join(frames_proc_dir, f"frame_{frame_idx:06d}.png")
            raw_frames_paths.append(raw_path)
            processed_frames_paths.append(proc_path)

            if not os.path.exists(raw_path):
                # Save the raw frame
                # We'll use Pillow
                pil_img = Image.fromarray(frame, mode="RGB")
                pil_img.save(raw_path)
        else:
            # not storing raw frames, just store them in memory
            raw_frames_paths.append(frame)  # store the actual array
            processed_frames_paths.append(None)  # marker

        frame_idx += 1

    video_reader.close()
    total_frames = frame_idx
    print(f"Total frames in video: {total_frames}")

    # 4) Process each frame with DETR
    print("Processing frames with DETR overlay...")
    for i in range(total_frames):
        if args.resumable:
            raw_path = raw_frames_paths[i]
            proc_path = processed_frames_paths[i]
            if os.path.exists(proc_path):
                # Already processed
                continue
            # Load PIL
            pil_img = Image.open(raw_path).convert("RGB")
            out_np = detect_and_overlay(model, pil_img, top_k=args.top_k)
            out_pil = Image.fromarray(out_np)
            out_pil.save(proc_path)
        else:
            # raw_frames_paths[i] is the actual numpy array
            frame_np = raw_frames_paths[i]
            pil_img = Image.fromarray(frame_np, mode="RGB")
            out_np = detect_and_overlay(model, pil_img, top_k=args.top_k)
            raw_frames_paths[i] = out_np  # store the processed version in memory

        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{total_frames} frames...")

    print("All frames processed.")

    # 5) Write output video with imageio
    print(f"Writing output video: {args.output_video_path} ...")
    video_writer = imageio.get_writer(args.output_video_path, fps=fps)

    for i in range(total_frames):
        if args.resumable:
            # read from frames_processed
            proc_path = processed_frames_paths[i]
            img_proc = Image.open(proc_path).convert("RGB")
            img_proc_np = np.array(img_proc)  # shape (H,W,3)
        else:
            # we stored the processed frames in memory
            img_proc_np = raw_frames_paths[i]
        video_writer.append_data(img_proc_np)

    video_writer.close()
    print(f"Done. Output video saved to {args.output_video_path}")


if __name__ == "__main__":
    main()
