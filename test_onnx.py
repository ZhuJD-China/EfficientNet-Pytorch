import onnx
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
# # Load the ONNX model
# model = onnx.load("alexnet.onnx")

# # Check that the model is well formed
# onnx.checker.check_model(model)

# # Print a human readable representation of the graph
# print(onnx.helper.printable_graph(model.graph))

import onnxruntime as ort

ort_session = ort.InferenceSession("/home/zjd/EfficientNet-Pytorch/efficientnet_b3_flower.onnx")

def get_key(dct, value):
    return [k for (k, v) in dct.items() if v == value]

data_transforms = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_dir = '/home/zjd/EfficientNet-Pytorch/OxFlower17/test/17/1348.jpg'
# load image
img = Image.open(img_dir)
inputs = data_transforms(img)
inputs.unsqueeze_(0)


outputs = ort_session.run(
    None,
    {"input": np.array(inputs)},
)

outputs = torch.tensor(outputs[0])
print(outputs)

_, preds = torch.max(outputs.data, 1)
# print(preds.item())
# print(mapping)
data = datasets.ImageFolder('OxFlower17/train')
mapping = data.class_to_idx
class_name = get_key(mapping, preds.item())
# use the mapping

print(img_dir)
print('prediction_label:', class_name)
print(30*'--')
