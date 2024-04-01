import torchinfo
import torch
import torch.nn as nn
import torchvision.models as models
from TNet import TNet

#model1 = HybridViT(num_classes=5, has_logits=False,init_weights=True)
model2 = TNet(num_classes=6)
# 设置input_size  因为会为你计算参数量

torchinfo.summary(model=model2,input_size=(1, 3, 224, 224))