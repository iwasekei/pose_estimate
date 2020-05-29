from model import convolutional_pose_machine_ind
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from cpm_utils import loader_py as ld
from cpm_utils import Traindataset_RE
from cpm_utils import Testdataset_RE
from cpm_utils import cmd_options

import log
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

    #prepare cpm_dataset and logger
    data = Traindataset_RE.Train(args.traindata_path, args.image_path, args.seg_path, args.im_size, args.rotate_range, args.scale_range, pairs)
    eva =  Testdataset_RE.Test(args.testdata_path, args.image_path, args.seg_path, args.im_size, args.pck_rate)
    log = log.Log(args.dir)
    
    #prepare u-net_dataset
    unet_train = ld.Loader(image_path=data.unet_im_path, seg_path=data.unet_se_path)
    unet_val = ld.Loader(image_path=eva.unet_im_path, seg_path=eva.unet_se_path)
    
    #gpu setting
    gpus = (args.gpu,)
    device = torch.device(f"cuda:{min(gpus)}" if len(gpus) > 0 else 'cpu')

    #prepare model
    VGG_model = convolutional_pose_machine_ind.VGG()
    UNet_model = convolutional_pose_machine_ind.UNet()
    CPM_model = convolutional_pose_machine_ind.CPM(args.n_point, args.n_stage)
    #model_load
    if args.vgg_init_model:
        VGG_model.load_state_dict(torch.load(args.vgg_init_model, map_location=device), strict=True)
    if args.unet_init_model:
        UNet_model.load_state_dict(torch.load(args.unet_init_model, map_location=device), strict=True)
    if args.cpm_init_model:
        CPM_model.load_state_dict(torch.load(args.cpm_init_model, map_location=device), strict=False)
        
    VGG_model.eval()
    UNet_model.eval()
    VGG_model.to(device)
    UNet_model.to(device)
    CPM_model.to(device)
    optimizer = optim.Adam(CPM_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = func)

    
    minloss = 1000.0
    
    #勾配情報保持のための値
    #dam = torch.tensor(0)
    #dam = dam.to(device)
    
    #trainloop
    for epoch in range(args.n_epoch):
        print('\n<epoch {}>'.format(epoch+1))
        perm = np.random.permutation(len(data))
        trainloss = 0
        testloss = 0
        
        #train
        print('training...')
        for i in tqdm(range(0, len(data), args.batchsize)):
            miniperm = perm[i:i+args.batchsize]
            #optimizer.zero_grad()
            
            #vgg train
            imgs = []
            labels = []#
            for l in range(len(miniperm)):
                img, label = data.vgg_generate(miniperm[l])#
                imgs += [img]
                labels += [label]#
            imgs = torch.from_numpy((np.transpose(np.array(imgs).astype(np.float32), (0,3,1,2))))
            imgs = imgs.to(device)
            #labels = torch.from_numpy(np.array(labels).astype(np.float32))#
            #labels = labels.to(device)
            #r_label = VGG_model.predict_vgg(imgs,labels)
            r_label = VGG_model.predict_vgg(imgs)
            
            #vgg_loss.backward(retain_graph=True)
            #optimizer.step()
            #vgg_loss = vgg_loss.to('cpu').data
            add_label = r_label.cpu().detach().clone().numpy()
            add_label = np.squeeze(np.argmax(add_label,axis=1))
            #trainloss += loss.to('cpu')*len(miniperm)
            
            #unet train
            inputs = []
            teacher = []
            for k in range(len(miniperm)):
                inputs += [unet_train._data.images_original[miniperm[k]]]
                teacher += [unet_train._data.images_segmented[miniperm[k]]]
            inputs_torch = torch.from_numpy((np.transpose(np.array(inputs).astype(np.float32), (0,3,1,2))))
            inputs_torch = inputs_torch.to(device)
            #teacher_torch = torch.from_numpy((np.transpose(np.array(teacher).astype(np.float32), (0,3,1,2))))
            #teacher_torch = teacher_torch.to(device)
            
            # Training
            #unet_h = UNet_model.predict_unet(inputs_torch, teacher_torch, dam)
            unet_h = UNet_model.predict_unet(inputs_torch)
            unet_h = unet_h.cpu().detach().clone().numpy()
            out = (np.transpose(np.array(unet_h).astype(np.float32), (0,2,3,1)))
            out = np.squeeze(np.argmax(out,axis=3))
            #unet_cpm.zero_grad()
            #unet_loss.backward(retain_graph=True)
            #optimizer.step()
            #unet_loss = unet_loss.to('cpu').data
            
            
            imgs = []
            b_maps = []
            c_maps = []
            labels = []

            for j in range(len(miniperm)):
                img, b_map, c_map = data.generate(miniperm[j], out[j], add_label[j])#
                imgs += [img]
                b_maps += [b_map]
                c_maps += [[c_map]]
                #labels += [label]#
                #print(labels)
            imgs = torch.from_numpy((np.transpose(np.array(imgs).astype(np.float32), (0,3,1,2))))
            imgs = imgs.to(device)
            b_maps = torch.from_numpy(np.array(b_maps).astype(np.float32))
            b_maps = b_maps.to(device)
            c_maps = torch.from_numpy(np.array(c_maps).astype(np.float32))
            c_maps = c_maps.to(device)
            #labels = torch.from_numpy(np.array(labels).astype(np.float32))#
            #labels = labels.to(device)
            h, cpm_loss = CPM_model.cpm(imgs, c_maps, b_maps, r_label)
            optimizer.zero_grad()#
            cpm_loss.backward()
            optimizer.step()

            trainloss += cpm_loss.to('cpu').data*len(miniperm)

        #test
        print('testing...')
        results = []
        for i in tqdm(range(len(eva))):
            #vgg
            img, label = eva.vgg_generate(i)
            img = img[np.newaxis,:,:,:]
            img = torch.from_numpy(np.transpose(np.array(img).astype(np.float32), (0,3,1,2)))
            img = img.to(device)
            #label = torch.from_numpy(np.array(label).astype(np.float32))#
            #label = label.to(device)
            r_label = VGG_model.predict_vgg(img)#
            #vgg_val_loss = loss.to('cpu').data
            add_label = r_label.cpu().detach().clone().numpy()
            add_label = np.squeeze(np.argmax(add_label,axis=1))
            
            
            #u-net
            inputs = unet_val._data.images_original[i]
            teacher = unet_val._data.images_segmented[i]
            inputs = inputs[np.newaxis]
            teacher = teacher[np.newaxis]
            inputs_torch = torch.from_numpy((np.transpose(np.array(inputs).astype(np.float32), (0,3,1,2))))
            inputs_torch = inputs_torch.to(device)
            #teacher_torch = torch.from_numpy((np.transpose(np.array(teacher).astype(np.float32), (0,3,1,2))))
            #teacher_torch = teacher_torch.to(device)
            
            h2 = UNet_model.predict_unet(inputs_torch)
            #unet_val_loss = loss.to('cpu').data
            h2 = h2.cpu().detach().clone().numpy()
            out = (np.transpose(np.array(h2).astype(np.float32), (0,2,3,1)))
            out = np.squeeze(np.argmax(out,axis=3))
            
            #CPM
            img, b_map, c_map = eva.generate(i, out, add_label)
            img = img[np.newaxis,:,:,:]
            b_map = b_map[np.newaxis,:,:,:]
            c_map = c_map[np.newaxis,np.newaxis,:,:]
            img = torch.from_numpy(np.transpose(np.array(img).astype(np.float32), (0,3,1,2)))
            img = img.to(device)
            b_map = torch.from_numpy(np.array(b_map).astype(np.float32))
            b_map = b_map.to(device)
            c_map = torch.from_numpy(np.array(c_map).astype(np.float32))
            c_map = c_map.to(device)
            #label = torch.from_numpy(np.array(label).astype(np.float32))#
            #label = label.to(device)
            h, loss = CPM_model.cpm(img, c_map, b_map,r_label)#
            cpm_val_loss = loss.to('cpu').data
            h2 = h.cpu().detach().clone().numpy()
            testloss += cpm_val_loss
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
            torch.save(CPM_model.state_dict(), os.path.join(args.dir,'trained_model','{0}'.format(args.gpu)))
