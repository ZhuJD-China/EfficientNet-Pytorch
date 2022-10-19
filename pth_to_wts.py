import torch
from torch import nn
#load你的模型
import os
import struct

net = torch.load('/home/zhujd/EfficientNet-Pytorch/OxFlower17/model/efficientnet-b3.pth') #loadpth文件
net = net.to('cuda:0')
net.eval()

f = open("efficientnet_b3_flowers.wts", 'w') #自己命名wts文件
f.write("{}\n".format(len(net.state_dict().keys())))  #保存所有keys的数量
for k,v in net.state_dict().items():
    #print('key: ', k)
    #print('value: ', v.shape)
    vr = v.reshape(-1).cpu().numpy()
    f.write("{} {}".format(k, len(vr)))  #保存每一层名称和参数长度
    for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())  #使用struct把权重封装成字符串
    f.write("\n")


