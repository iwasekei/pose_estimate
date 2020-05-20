import torch
#from chainer import Variable
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.links as L
#from model.se_module import Pfilter #
from util import loader as ld

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
        self.conv_out=nn.Conv2d(64, ld.DataSet.length_category(), 1, stride=1, padding=0)

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
        #print(h5_t.shape)
        h5 = torch.cat((h5_t, h4_2), 1)
        #print(h5.shape)
        
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
        #out = self.conv_out(hup4_2)
        out = F.softmax(self.conv_out(hup4_2), dim=1)
        
        return out

class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        self.model = Model()
        #self.add_link('branch', Branch())
        #self.add_link('stage1', Stage1(n_point))
        #links = []
        #branches=[]
        #for i in range(n_stage-1):
        #    links += [('stage{}'.format(i+2), StageN(n_point))]
            #branches+=[('branch{}'.format(i+2), Branch())]
        #for l in links:
        #    self.add_link(*l)
        #for b in branches:
        #    self.add_link(*b)

        #self.n_stage = n_stage
        #self.forward = links
        #self.branches=branches
        #self.train = True

    #def clear(self):
    #    self.loss = None

    def __call__(self, inputs, target):
        loss = None
        #criterion = nn.NLLLoss2d()
        h = self.model(inputs)
        #h1 = F.dropout(h1)#
        #h2 = self.branch(image,label)
        #print(h1.shape,t.shape)
        #print(h.shape, teacher.shape)
        #print(h.shape)
        
        loss = torch.sum(- target * F.log_softmax(h, -1), -1)
        mean_loss = loss.mean()

        #for i in range(self.n_stage-1):
        #    stage_name='stage{}'.format(i+2)
        #    branch_name='branch{}'.format(i+2)#
        #    f = getattr(self, stage_name)
        #    b_f = getattr(self, branch_name)
        #    h2 = b_f(image)
        #    #h2 = F.dropout(h2)
        #    h1 = f(h1, h2, cmap,label)#
        #    #h1 = F.dropout(h1)#
        #    self.loss += F.mean_squared_error(h1, t)

        #for name, _ in self.forward:
        #    f = getattr(self, name)
        #    h1 = f(h1, h2, cmap,label)#
        #    loss += F.mse_loss(h1, t)

        return h, mean_loss

    def predict(self, inputs):#
        h = self.model(inputs)#
        #h1 = F.dropout(h1)
        #for name, _ in self.forward:
        #    f = getattr(self, name)
        #    h1 = f(h1, h2, cmap,label)#
        #for i in range(self.n_stage-1):
        #               
        #    stage_name='stage{}'.format(i+2)
        #    branch_name='branch{}'.format(i+2)#
        #    f = getattr(self, stage_name)
        #    b_f = getattr(self, branch_name)
        #    h2 = b_f(image)
        #    #h2 = F.dropout(h2)
        #    h1 = f(h1, h2, cmap,label)#
        #    #h1 = F.dropout(h1)#
        #    #self.loss += F.mean_squared_error(h1, t)
        return h
