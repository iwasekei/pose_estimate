import argparse
#import tensorflow as tf
import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm
import math
import os

from unet_utils import loader as ld
from unet_utils import repoter as rp
from model import UNet


def load_dataset(train_rate, pattern):
    loader = ld.Loader(dir_original="../data_set/images",
                       dir_segmented="../data_set/seg")
    return loader.load_train_test(train_rate=train_rate, shuffle=False, pattern=pattern)


def train(parser):
    # Load train and test datas
    train, train_test, val, test = load_dataset(train_rate=parser.trainrate, pattern=parser.pattern)

    # Create Reporter Object
    reporter = rp.Reporter(parser=parser)
    accuracy_fig = reporter.create_figure("Accuracy", ("epoch", "accuracy"), ["train", "test"])
    loss_fig = reporter.create_figure("Loss", ("epoch", "loss"), ["train", "test"])

    # Whether or not using a GPU
    gpus = (parser.gpu,)
    device = torch.device(f"cuda:{min(gpus)}" if len(gpus) > 0 else 'cpu')

    # Create a model
    model_unet = UNet.UNet()
    model_unet.to(device)

    # Set optimizer
    optimizer = optim.Adam(model_unet.parameters(), lr=0.001, weight_decay=parser.l2reg)

    
    # Train the model
    epochs = parser.epoch
    batch_size = parser.batchsize
    is_augment = parser.augmentation

    minloss = 10000
    for epoch in range(epochs):
        print('\n<epoch {}>'.format(epoch+1))
        perm = np.random.permutation(len(train.images_original))
        trainloss = 0
        testloss = 0
        
        print('training...')
        for batch in tqdm(train(batch_size=batch_size, augment=is_augment)):
            inputs = batch.images_original
            teacher = batch.images_segmented
            inputs_torch = torch.from_numpy((np.transpose(np.array(inputs).astype(np.float32), (0,3,1,2))))
            inputs_torch = inputs_torch.to(device)
            teacher_torch = torch.from_numpy((np.transpose(np.array(teacher).astype(np.float32), (0,3,1,2))))
            teacher_torch = teacher_torch.to(device)
            
            h, loss = model_unet(inputs_torch, teacher_torch)
            
            model_unet.zero_grad()
            loss.backward()
            optimizer.step()
            
            trainloss += loss.to('cpu').data*len(batch.images_original)

        print('testing...')
        results = []
        for i in tqdm(range(len(val.images_original))):
            inputs = val.images_original[i]
            teacher = val.images_segmented[i]
            inputs = inputs[np.newaxis]
            teacher = teacher[np.newaxis]
            inputs_torch = torch.from_numpy((np.transpose(np.array(inputs).astype(np.float32), (0,3,1,2))))
            inputs_torch = inputs_torch.to(device)
            teacher_torch = torch.from_numpy((np.transpose(np.array(teacher).astype(np.float32), (0,3,1,2))))
            teacher_torch = teacher_torch.to(device)
            
            h2, loss = model_unet(inputs_torch, teacher_torch)
            testloss += loss.to('cpu').data
            h2 = h2.cpu().detach().clone().numpy()
            
            out = (np.transpose(np.array(h2).astype(np.float32), (0,2,3,1)))
            res = np.argmax(out, axis=3)
            tea = np.argmax(np.array(teacher), axis=3)
            diff = (np.sum(np.abs(res-tea)))
            result = 1-(diff/(128**2))
            results += [result]
            
        print('trainloss = {}'.format(trainloss/len(train.images_original)))
        print('valloss = {}'.format(testloss/len(val.images_original)))
        print('valaccu = {}'.format(np.array(results).mean()))
        loss_fig.add([trainloss/len(train.images_original), testloss/len(val.images_original)], is_update=True)
                    
        if minloss>(testloss/len(val.images_original)):
            minloss = testloss/len(val.images_original)
            reporter.save_loss_model(model_unet, parser.gpu)


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-g', '--gpu', type=int, default=0, help='Using GPUs')
    parser.add_argument('-e', '--epoch', type=int, default=250, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float, default=0.875, help='Training rate')
    parser.add_argument('-a', '--augmentation', action='store_true', help='Number of epochs')
    parser.add_argument('-r', '--l2reg', type=float, default=0.0001, help='L2 regularization')
    parser.add_argument('--pattern', type=int, default=0, help='data pattern')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
