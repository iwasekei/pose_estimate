import torch
#from chainer import Variable
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.links as L
#from model.se_module import Pfilter #
from torchvision import models

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=False)
        for p in self.vgg16.features.parameters():
            p.requires_grad = True
        self.vgg16.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 4)
        )
        
    def __call__(self, x):
        h = self.vgg16(x)
        h = F.softmax(h, dim=1)
        
        return h
    
class Class(nn.Module):

    def __init__(self):
        super(Class, self).__init__()
        self.vgg = VGG16()
        
    def __call__(self, x):
        h = self.vgg(x)
        
        return h
    
    def predict_vgg(self, image, cmap, label):#
        h1 = self.stage1(image,label)#
        h2 = self.branch(image,label)
        h1 = self.stage2(h1, h2, cmap,label)#
        h1 = self.stage3(h1, h2, cmap,label)#
        h1 = self.stage4(h1, h2, cmap,label)#
        h1 = self.stage5(h1, h2, cmap,label)#
        h1 = self.stage6(h1, h2, cmap,label)#
        return h1
