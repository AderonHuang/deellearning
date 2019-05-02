#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:16:57 2019

@author: aderon
"""

import torch
import torch.nn as nn
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        layer1 = nn.Sequential()
        #Conv2d(inchannel，outchannel，kenel_size)
        layer1.add_module('conv1',nn.Conv2d(1,6,5))
        layer1.add_module('pool1',nn.MaxPool2d(2,2))
        self.layer1 = layer1
        
        layer2 = nn.Sequential()
        layer2.add_module('conv2',nn.Conv2d(6,16,5))
        layer2.add_module('pool2',nn.MaxPool2d(2,2))
        self.layer2 = layer2
        
        layer3 = nn.Sequential()
        layer3.add_module('fc1',nn.Linear(16*5*5,120))
        layer3.add_module('fc2',nn.Linear(120,84))
        layer3.add_module('fc3',nn.Linear(84,10))
        self.layer3 = layer3
        
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0),-1)#降低数据维度，进入全连接层
        x = self.layer3(x)
        return x
#y = torch.randn(1,1,32,32)
#model = LeNet()
#result = model(y)
#print(result)