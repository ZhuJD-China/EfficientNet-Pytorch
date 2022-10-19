from __future__ import print_function, division

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

net_name = 'efficientnet-b3'

# load model
save_dir = 'OxFlower17/model'
modelft_file = save_dir + "/" + net_name + '.pth'

# use GPU
model = torch.load(modelft_file).cuda()
model.eval()

# export onnx
batch_size = 1  #批处理大小
input_shape = (3, 300, 300)   #输入数据,改成自己的输入shape
x = torch.randn(batch_size, *input_shape).cuda()  # 生成张量
export_onnx_file = "efficientnet_b3_flower.onnx"          # 目的ONNX文件名
model.set_swish(memory_efficient=False)
torch.onnx.export(model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,  # 是否执行常量折叠优化
                    input_names=["input"], # 输入名
                    output_names=["output"])   # 输出名
