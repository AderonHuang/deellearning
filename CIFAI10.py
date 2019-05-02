#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 22:51:46 2019

@author: aderon
"""
import torch as t
import torchvision as tv  
import torchvision.transforms as transforms  
from torchvision.transforms import ToPILImage  
#show = ToPILImage() # 可以把Tensor转成Image，方便可视化  
# 第一次运行程序torchvision会自动下载CIFAR-10数据集，  
# 大约100M，需花费一定的时间，  
# 如果已经下载有CIFAR-10，可通过root参数指定  
  
# 定义对数据的预处理  
transform = transforms.Compose([  
transforms.ToTensor(), # 转为Tensor  
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化  
])  
  
# 训练集  
trainset = tv.datasets.CIFAR10(  
root='/home/aderon/.config/spyder-py3/basenet/Lenet/cifar-10-python/',  
train=True,  
download=True,  
transform=transform)  
  
trainloader = t.utils.data.DataLoader(  
trainset,  
batch_size=4,  
shuffle=True,  
num_workers=2)  
  
# 测试集  
testset = tv.datasets.CIFAR10(  
'/home/aderon/.config/spyder-py3/basenet/Lenet/cifar-10-python/',  
train=False,  
download=True,  
transform=transform)  
  
testloader = t.utils.data.DataLoader(  
testset,  
batch_size=4,  
shuffle=False,  
num_workers=2)  
  
classes = ('plane', 'car', 'bird', 'cat',  
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  