from model import convolutional_pose_machine_FW
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

import numpy as np
from cpm_utils import Traindataset
from cpm_utils import Testdataset
from cpm_utils import cmd_options
from cpm_utils import log
import sys
import csv
import os
pairs = ((12,13),(11,14),(10,15),(2,3),(1,4), (0,5))

def func(epoch):
    if epoch < 100:
        return 0.001
    else:
        return 0.001**2

if __name__ == '__main__' :
    args = cmd_options.get_arguments()

    #prepare dataset and logger
    data = Traindataset.Train(args.traindata_path, args.image_path, args.im_size,
                         args.rotate_range, args.scale_range, pairs)
    eva =  Testdataset.Test(args.testdata_path, args.image_path, args.im_size, args.pck_rate)
    log = log.Log(args.dir)
    
    #set_gpu
    gpus = (args.gpu,)
    device = torch.device(f"cuda:{min(gpus)}" if len(gpus) > 0 else 'cpu')

    #prepare model
    cpm = convolutional_pose_machine_FW.CPM(args.n_point, args.n_stage)
    #load_model
    if args.init_model:
        cpm.load_state_dict(torch.load(args.init_model, map_location=device), strict=False)

    cpm.to(device)

    optimizer = optim.Adam(cpm.parameters(), lr=args.lr)
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
            imgs = []
            b_maps = []
            c_maps = []
            labels = []#
            miniperm = perm[i:i+args.batchsize]

            for j in range(len(miniperm)):
                img, b_map, c_map, label = data.generate(miniperm[j])#
                imgs += [img]
                b_maps += [b_map]
                c_maps += [[c_map]]
                labels += [label]#
            imgs = torch.from_numpy((np.transpose(np.array(imgs).astype(np.float32), (0,3,1,2))))
            imgs = imgs.to(device)
            b_maps = torch.from_numpy(np.array(b_maps).astype(np.float32))
            b_maps = b_maps.to(device)
            c_maps = torch.from_numpy(np.array(c_maps).astype(np.float32))
            c_maps = c_maps.to(device)
            labels = torch.from_numpy(np.array(labels).astype(np.float32))#
            labels = labels.to(device)
            h, loss = cpm(imgs, c_maps, b_maps, labels)

            cpm.zero_grad()
            loss.backward()
            optimizer.step()

            trainloss += loss.to('cpu')*len(miniperm)

        #test
        print('testing...')
        results = []
        for i in tqdm(range(len(eva))):
            img, b_map, c_map, label = eva.generate(i)
            img = img[np.newaxis,:,:,:]
            b_map = b_map[np.newaxis,:,:,:]
            c_map = c_map[np.newaxis,np.newaxis,:,:]
            label = label[np.newaxis]
            img = torch.from_numpy(np.transpose(np.array(img).astype(np.float32), (0,3,1,2)))
            img = img.to(device)
            b_map = torch.from_numpy(np.array(b_map).astype(np.float32))
            b_map = b_map.to(device)
            c_map = torch.from_numpy(np.array(c_map).astype(np.float32))
            c_map = c_map.to(device)
            label = torch.from_numpy(np.array(label).astype(np.float32))#
            label = label.to(device)
            h, loss = cpm(img, c_map, b_map,label)#
            testloss += loss.data
            h2 = h.cpu().detach().clone().numpy()
            result = eva.evaluate(i, h2[0])
            results +=[result]

        results = np.array(results)
        print ('trainloss = {}'.format(trainloss / len(data)))
        print ('valloss = {}'.format(testloss / len(eva)))
        print ('valacc  = {}'.format(results.mean()))
        # log and save
        log(epoch+1, results.mean(), trainloss/len(data), testloss/len(eva))
        if minloss>(testloss/len(eva)):
            minloss = testloss/len(eva)
            torch.save(cpm.state_dict(), os.path.join(args.dir,'trained_model','{0}'.format(args.gpu)))
