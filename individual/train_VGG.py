from model import VGG
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from vgg_utils import Traindataset_vgg
from vgg_utils import Testdataset_vgg
from vgg_utils import cmd_options
from vgg_utils import log
import sys
import csv
import os
from torchvision import models
pairs = ((12,13),(11,14),(10,15),(2,3),(1,4), (0,5))

def func(epoch):
    if epoch < 100:
        return 0.005
    else:
        return 0.005**2

if __name__ == '__main__' :
    args = cmd_options.get_arguments()

    #prepare dataset and logger
    data = Traindataset_vgg.Train(args.traindata_path, args.image_path, args.seg_path, args.im_size, args.rotate_range, args.scale_range, pairs)
    eva =  Testdataset_vgg.Test(args.testdata_path, args.image_path, args.seg_path, args.im_size, args.pck_rate)
    log = log.Log(args.dir)
    
    
    gpus = (args.gpu,)
    device = torch.device(f"cuda:{min(gpus)}" if len(gpus) > 0 else 'cpu')
    
    #model load
    vgg16 = VGG.Class()
    vgg16.to(device)
    
    optimizer = optim.Adam(vgg16.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = func)

    #trainloop
    minloss = 1000.0
    for epoch in range(args.n_epoch):
        print('\n<epoch {}>'.format(epoch+1))
        perm = np.random.permutation(len(data))
        trainloss = 0
        testloss = 0
        
        #train
        print('training...')
        for i in tqdm(range(0, len(data), args.batchsize)):
            miniperm = perm[i:i+args.batchsize]
            inputs = []
            teacher = []
            
            imgs = []
            labels = []

            for j in range(len(miniperm)):
                img, label = data.vgg_generate(miniperm[j])
                imgs += [img]
                labels += [label]#
            imgs = torch.from_numpy((np.transpose(np.array(imgs).astype(np.float32), (0,3,1,2))))
            imgs = imgs.to(device)
            labels = torch.from_numpy(np.array(labels).astype(np.float32))#
            labels = labels.to(device)
            h = vgg16(imgs)
            loss = torch.sum(- labels * F.log_softmax(h, -1), -1)
            loss = loss.mean()
            vgg16.zero_grad()
            loss.backward()
            optimizer.step()

            trainloss += loss.to('cpu')*len(miniperm)

        #test
        print('testing...')
        results = 0
        for i in tqdm(range(len(eva))):
            
            img, label = eva.vgg_generate(i)
            img = img[np.newaxis,:,:,:]
            img = torch.from_numpy(np.transpose(np.array(img).astype(np.float32), (0,3,1,2)))
            img = img.to(device)
            label = torch.from_numpy(np.array(label).astype(np.float32))#
            label = label.to(device)
            h = vgg16(img)#
            loss = torch.sum(- label * F.log_softmax(h, -1), -1)
            loss = loss.mean()
            cpm_val_loss = loss.to('cpu').data
            testloss += cpm_val_loss
            h = h.cpu().detach().clone().numpy()
            res = np.argmax(h, axis=1)
            tea = np.argmax(np.array(label.cpu().detach().clone().numpy()), axis=1)
            if res-tea == 0:
                results += 1

        results = np.array(results)
        print ('trainloss = {}'.format(trainloss / len(data)))
        print ('valloss = {}'.format(testloss / len(eva)))
        print ('valacc  = {}'.format(results/len(eva)))
        # log and save
        log(epoch+1, results.mean(), trainloss/len(data), testloss/len(eva))
        if minloss>(testloss/len(eva)):
            minloss = testloss/len(eva)
            torch.save(vgg16.state_dict(), os.path.join(args.dir,'trained_model','{0}'.format(args.gpu)))