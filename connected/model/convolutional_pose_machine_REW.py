import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.se_module import Pfilter #

#-------------------
#model定義部分は変更なし
#line242~ Class CPM_RWより変更
#-------------------

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

class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.conv1_1=nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1_1=nn.BatchNorm2d(64)
        self.conv1_2=nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn1_2=nn.BatchNorm2d(64)
        
        self.conv2_1=nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn2_1=nn.BatchNorm2d(128)
        self.conv2_2=nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2_2=nn.BatchNorm2d(128)
        
        self.conv3_1=nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3_1=nn.BatchNorm2d(256)
        self.conv3_2=nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn3_2=nn.BatchNorm2d(256)
        
        self.conv4_1=nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn4_1=nn.BatchNorm2d(512)
        self.conv4_2=nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn4_2=nn.BatchNorm2d(512)
        
        self.conv5_1=nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.conv5_2=nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.conv_transpose5=nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0)
        
        self.conv_up1_1=nn.Conv2d(1024, 512, 3, stride=1, padding=1)
        self.conv_up1_2=nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_transpose1=nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)
        
        self.conv_up2_1=nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.conv_up2_2=nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_transpose2=nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
        
        self.conv_up3_1=nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv_up3_2=nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_transpose3=nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
        
        self.conv_up4_1=nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv_up4_2=nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_out=nn.Conv2d(64, 2, 1, stride=1, padding=0)
        
    def __call__(self, x):
        h1_1 = self.conv1_1(x)
        h1_1 = F.relu(h1_1)
        h1_1 = self.bn1_1(h1_1)
        h1_2 = self.conv1_2(h1_1)
        h1_2 = F.relu(h1_2)
        h1_2 = self.bn1_2(h1_2)
        h1 = F.max_pool2d(h1_2, 2, stride=2, padding=0)
        
        h2_1 = self.conv2_1(h1)
        h2_1 = F.relu(h2_1)
        h2_1 = self.bn2_1(h2_1)
        h2_2 = self.conv2_2(h2_1)
        h2_2 = F.relu(h2_2)
        h2_2 = self.bn2_2(h2_2)
        h2 = F.max_pool2d(h2_2, 2, stride=2, padding=0)
        
        h3_1 = self.conv3_1(h2)
        h3_1 = F.relu(h3_1)
        h3_1 = self.bn3_1(h3_1)
        h3_2 = self.conv3_2(h3_1)
        h3_2 = F.relu(h3_2)
        h3_2 = self.bn3_2(h3_2)
        h3 = F.max_pool2d(h3_2, 2, stride=2, padding=0)
        
        h4_1 = self.conv4_1(h3)
        h4_1 = F.relu(h4_1)
        h4_1 = self.bn4_1(h4_1)
        h4_2 = self.conv4_2(h4_1)
        h4_2 = F.relu(h4_2)
        h4_2 = self.bn4_2(h4_2)
        h4 = F.max_pool2d(h4_2, 2, stride=2, padding=0)
        
        h5_1 = self.conv5_1(h4)
        h5_1 = F.relu(h5_1)
        h5_2 = self.conv5_2(h5_1)
        h5_2 = F.relu(h5_2)
        h5_t = self.conv_transpose5(h5_2)
        h5_t = F.relu(h5_t)
        h5 = torch.cat((h5_t, h4_2), 1)
        
        hup1_1 = self.conv_up1_1(h5)
        hup1_1 = F.relu(hup1_1)
        hup1_2 = self.conv_up1_2(hup1_1)
        hup1_2 = F.relu(hup1_2)
        hup1_t = self.conv_transpose1(hup1_2)
        hup1_t = F.relu(hup1_t)
        hup1 = torch.cat((hup1_t, h3_2), 1)
        
        hup2_1 = self.conv_up2_1(hup1)
        hup2_1 = F.relu(hup2_1)
        hup2_2 = self.conv_up2_2(hup2_1)
        hup2_2 = F.relu(hup2_2)
        hup2_t = self.conv_transpose2(hup2_2)
        hup2_t = F.relu(hup2_t)
        hup2 = torch.cat((hup2_t, h2_2), 1)
        
        hup3_1 = self.conv_up3_1(hup2)
        hup3_1 = F.relu(hup3_1)
        hup3_2 = self.conv_up3_2(hup3_1)
        hup3_2 = F.relu(hup3_2)
        hup3_t = self.conv_transpose3(hup3_2)
        hup3_t = F.relu(hup3_t)
        hup3 = torch.cat((hup3_t, h1_2), 1)
        
        hup4_1 = self.conv_up4_1(hup3)
        hup4_1 = F.relu(hup4_1)
        hup4_2 = self.conv_up4_2(hup4_1)
        hup4_2 = F.relu(hup4_2)
        out = F.softmax(self.conv_out(hup4_2), dim=1)
        
        return out

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

class CPM_RW(nn.Module):

    def __init__(self, n_point, n_stage):
        super(CPM_RW, self).__init__()
        self.vgg = VGG16()
        self.model = Model()
        self.branch = Branch()
        self.stage1 = Stage1(n_point)
        self.stage2 = StageN(n_point)
        self.stage3 = StageN(n_point)
        self.stage4 = StageN(n_point)
        self.stage5 = StageN(n_point)
        self.stage6 = StageN(n_point)
        
    def vgg16(self, input, target):
        h = self.vgg(input)#
        loss = torch.sum(- target * F.log_softmax(h, -1), -1)
        loss = loss.mean()
        
        return h, loss
    
    def unet(self, inputs, target, dam):
        loss = None
        h = self.model(inputs)
        
        loss = torch.sum(- target * F.log_softmax(h, -1), -1)
        mean_loss = loss.mean()
        dam_unet = torch.sum(h) * dam
        
        return h, mean_loss, dam_unet

    def cpm(self, image, cmap, t,label, dam):
        loss = None
        image = image + dam
        h1 = self.stage1(image,label)
        h2 = self.branch(image,label)
        loss = F.mse_loss(h1, t)
        
        h1 = self.stage2(h1, h2, cmap,label)#
        loss += F.mse_loss(h1, t)
        h1 = self.stage3(h1, h2, cmap,label)#
        loss += F.mse_loss(h1, t)
        h1 = self.stage4(h1, h2, cmap,label)#
        loss += F.mse_loss(h1, t)
        h1 = self.stage5(h1, h2, cmap,label)#
        loss += F.mse_loss(h1, t)
        h1 = self.stage6(h1, h2, cmap,label)#
        loss += F.mse_loss(h1, t)
        

        return h1, loss
    
    def predict_vgg(self, inputs):#
        h = self.vgg(inputs)#
        return h
    
    def predict_unet(self, inputs):#
        h = self.model(inputs)#
        return h
    
    def predict_cpm(self, image, cmap, label):#
        h1 = self.stage1(image,label)#
        h2 = self.branch(image,label)
        h1 = self.stage2(h1, h2, cmap,label)#
        h1 = self.stage3(h1, h2, cmap,label)#
        h1 = self.stage4(h1, h2, cmap,label)#
        h1 = self.stage5(h1, h2, cmap,label)#
        h1 = self.stage6(h1, h2, cmap,label)#
        return h1
