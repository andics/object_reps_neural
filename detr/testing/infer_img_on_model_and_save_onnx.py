import torch
import torchvision.models
import torchvision.transforms as T
import time

from PIL import Image
torch.set_grad_enabled(False)

MODEL_PATH = "/home/projects/bagon/andreyg/Projects/Variable_Resolution_DETR/Programming/detr_var/trained_models/variable_pretrained_resnet101/box_and_segm/checkpoint.pth"
ONNX_PATH = "/home/projects/bagon/andreyg/Projects/Variable_Resolution_DETR/Programming/detr_var/trained_models/variable_pretrained_resnet101/box_and_segm/detr_var_arch.onnx"

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#---MODEL_LOADING---
model = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', num_classes=91, pretrained=False)
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model"])

# Put the model in evaluation mode
model.eval()

#-------------------
im = Image.open("/home/projects/bagon/andreyg/Projects/Variable_Resolution_DETR/Datasets/coco_variable/val2017/000000039769.jpg")

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)

# propagate through the model
outputs = model(img)

# Export the model
torch.onnx.export(model,               # model being run
                  img,                         # model input (or a tuple for multiple inputs)
                  ONNX_PATH,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['pred_logits', 'pred_boxes', 'pred_masks'], # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},
                                'pred_logits': {0: 'batch_size'},
                                'pred_boxes': {0: 'batch_size'},
                                'pred_masks': {0, 'batch_size'}})
