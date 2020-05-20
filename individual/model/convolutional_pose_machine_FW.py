import torch
import torch.nn as nn
import torch.nn.functional as F
from model.se_module import Pfilter #

class Stage1(nn.Module):
    
    def __init__(self, n_point):
        super(Stage1, self).__init__()
        self.conv1=nn.Conv2d(3, 128, 9, stride=1, padding=4)
        self.conv2=nn.Conv2d(128, 128, 9, stride=1, padding=4)
        self.conv3=nn.Conv2d(128, 128, 9, stride=1, padding=4)
        self.conv4=nn.Conv2d(128, 32, 5, stride=1, padding=2)
        self.pfilter = Pfilter(32) #pose inf#
        self.conv5=nn.Conv2d(32, 512, 9, stride=1, padding=4)#
        self.conv6=nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv7=nn.Conv2d(512, n_point+1, 1, stride=1, padding=0)

    def __call__(self, x, label):
        h = self.conv1(x)
        h = F.relu(h)
        h = F.max_pool2d(h, 3, stride=2, padding=1)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pool2d(h, 3, stride=2, padding=1)
        h = self.conv3(h)
        h = F.relu(h)
        h = F.max_pool2d(h, 3, stride=2, padding=1)
        h = self.conv4(h) 
        h = F.relu(h)
        h = self.pfilter(h,label)#
        h = self.conv5(h)
        h = F.relu(h)
        h = self.conv6(h)
        h = F.relu(h)
        h = self.conv7(h)
        
        return h

class Branch(nn.Module):

    def __init__(self):
        super(Branch, self).__init__()
        self.conv1=nn.Conv2d(3, 128, 9, stride=1, padding=4)
        self.conv2=nn.Conv2d(128, 128, 9, stride=1, padding=4)
        self.conv3=nn.Conv2d(128, 128, 9, stride=1, padding=4)
        self.pfilter = Pfilter(128)#pose inf
        

    def __call__(self, x,label):
        h = self.conv1(x)
        h = F.relu(h)
        h = F.max_pool2d(h, 3, stride=2, padding=1)
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pool2d(h, 3, stride=2, padding=1)
        h = self.conv3(h)
        h = F.relu(h)
        h = F.max_pool2d(h, 3, stride=2, padding=1)
        h = self.pfilter(h,label)#
        return h

class StageN(nn.Module):

    def __init__(self, n_point):
        super(StageN, self).__init__()
        self.conv0=nn.Conv2d(128, 32, 5, stride=1, padding=2)
        self.conv1=nn.Conv2d(32 + (n_point+1) + 1, 128, 11, stride=1, padding=5)
        self.conv2=nn.Conv2d(128, 128, 11, stride=1, padding=5)
        self.conv3=nn.Conv2d(128, 128, 11, stride=1, padding=5)
        self.conv4=nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv5=nn.Conv2d(128, n_point+1, 1, stride=1, padding=0)

    def __call__(self, pmap, fmap, cmap,label):
        fmap = self.conv0(fmap)
        fmap = F.relu(fmap)
        cmap = F.avg_pool2d(cmap, 8, stride=8)

        h = torch.cat((fmap, pmap, cmap), 1)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = self.conv3(h)
        h = F.relu(h)
        h = self.conv4(h)
        h = F.relu(h)
        h = self.conv5(h)

        return h

class CPM(nn.Module):

    def __init__(self, n_point, n_stage):
        super(CPM, self).__init__()
        self.branch = Branch()
        self.stage1 = Stage1(n_point)
        self.stage2 = StageN(n_point)
        self.stage3 = StageN(n_point)
        self.stage4 = StageN(n_point)
        self.stage5 = StageN(n_point)
        self.stage6 = StageN(n_point)

    def __call__(self, image, cmap, t,label):
        loss = None
        h1 = self.stage1(image,label)
        h2 = self.branch(image,label)
        loss = F.mse_loss(h1, t)

        h1 = self.stage2(h1, h2, cmap,label)
        loss += F.mse_loss(h1, t)
        h1 = self.stage3(h1, h2, cmap,label)
        loss += F.mse_loss(h1, t)
        h1 = self.stage4(h1, h2, cmap,label)
        loss += F.mse_loss(h1, t)
        h1 = self.stage5(h1, h2, cmap,label)
        loss += F.mse_loss(h1, t)
        h1 = self.stage6(h1, h2, cmap,label)
        loss += F.mse_loss(h1, t)

        return h1, loss

    def predict(self, image, cmap, label):#
        h1 = self.stage1(image,label)#
        h2 = self.branch(image,label)
        h1 = self.stage2(h1, h2, cmap,label)#
        h1 = self.stage3(h1, h2, cmap,label)#
        h1 = self.stage4(h1, h2, cmap,label)#
        h1 = self.stage5(h1, h2, cmap,label)#
        h1 = self.stage6(h1, h2, cmap,label)#

        return h1
