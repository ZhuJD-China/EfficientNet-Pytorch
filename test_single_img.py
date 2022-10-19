# from __future__ import print_function, division

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import torch.nn as nn
import numpy as np

data_transforms = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_key(dct, value):
    return [k for (k, v) in dct.items() if v == value]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# find the mapping of folder-to-label
data = datasets.ImageFolder('OxFlower17/train')
mapping = data.class_to_idx
# print(mapping)

# start testing
net_name = 'efficientnet-b3'
img_dir = '/home/zjd/EfficientNet-Pytorch/OxFlower17/test/17/1348.jpg'

# load model
save_dir = 'OxFlower17/model'
modelft_file = save_dir + "/" + net_name + '.pth'

# load image
img = Image.open(img_dir)
inputs = data_transforms(img)
inputs.unsqueeze_(0)

# use GPU
model = torch.load(modelft_file).cuda()
model.eval()
# use GPU
inputs = Variable(inputs.cuda())
print("inputs:",inputs.shape)

# forward
outputs = model(inputs)

print(outputs)

_, preds = torch.max(outputs.data, 1)

print(preds.item())
print(mapping)

newdict = dict(zip(mapping.values(),mapping.keys()))
print(newdict)
print(type(newdict))
# class_name = get_key(newdict, preds.item())
class_name = newdict.get(preds.item())
# use the mapping

print(img_dir)
print('prediction_label:', class_name)
# print('prediction_label:', class_name," cofidence:",outputs[preds])
print(30*'--')
