import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
from transformers import SamModel, SamProcessor

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# Load image
img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# Input point
input_points = [[[450, 600]]]

# Prepare input
inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Post-process masks → shape: [3, 1, H, W]
processed_masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu()
)[0]

# Get IoU scores: shape [1, 1, 3]
scores = outputs.iou_scores.squeeze()  # now shape: [3]
best_idx = scores.argmax().item()
score = scores[best_idx].item()

# Extract the best mask: shape [1, H, W] → [H, W]
best_mask = processed_masks[best_idx][0].numpy()

# Plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(raw_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(raw_image)
plt.imshow(best_mask, alpha=0.5, cmap="jet")
plt.title(f"Best Mask (IoU: {score:.2f})")
plt.axis("off")

plt.tight_layout()
plt.show()
