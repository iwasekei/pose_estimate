from model import convolutional_pose_machine_REW
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from cpm_util import loader_py as ld
from cpm_util import log
from PIL import Image


import matplotlib as mpl
mpl.use('Agg')

import argparse
import copy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
import os
from tqdm import tqdm
import math

pose = ('r-ankle','r-knee','r-hip', 'l-hip', 'l-knee', 'l-ankle', 'pelvis', 'thorax',
        'upperneck', 'head top', 'r-wrist', 'r-elbow', 'r-shoulder', 'l-shoulder', 'l-elbow', 'l-wrist')
color = ((18,0,230),(0,152,243),(0,241,225,),(31,195,143),(68,153,0),(150,158,0),(223,160,0),
         (183,104,0),(136,32,29),(131,7,146),(127,0,228),(79,0,229),(255,255,255),(141,149,0),
         (226,188,163),(105,168,105))

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='Set GPU device numbers with comma saparated. '
                        'Default is 0.')
    parser.add_argument('--n_stage', type=int, default=6,
                        help='Set the number of stages of your network '
                        'Default is 4 stages.')
    parser.add_argument('--cpm_init_model', type=str, default='trained_model.model',
                        help='Path to chainer model to load before trainig.')
    parser.add_argument('--unet_init_model', type=str, default='trained_model.model',
                        help='Path to chainer model to load before trainig.')
    parser.add_argument('--vgg_init_model', type=str, default='trained_model.model',
                        help='Path to chainer model to load before trainig.')
    parser.add_argument('--n_point', type=int, default=16,
                        help='Set the number of joint points')
    parser.add_argument('--im_size', type=int, default=368,
                        help='Set input size of your network')
    parser.add_argument('--csv', type=str, default='',
                        help='Set input pas of your csv file')
    parser.add_argument('--im_path', type=str, default='',
                        help='Set input pas of your img file')
    parser.add_argument('--seg_path', type=str, default='',
                    help='Set input pas of your img file')
    parser.add_argument('--seg_im_path', type=str, default='',
                    help='Set input pas of your img file')
    #parser.add_argument('--imagename', type=str, default='',
    #                   help='Set the name of image')
    #parser.add_argument('--resultname', type=str, default='',
     #                   help='Set the name of result image')
    parser.add_argument('--c_deviation', type=float, default=2.0,
                        help='Set deviation of center map')

    args = parser.parse_args()

    return args


def makeGaussian(size, xd, yd, center, order=1000):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    x0 = center[0]
    y0 = center[1]
    return order*np.exp(-0.5*(((x-x0)**2)/xd**2 + ((y-y0)**2)/yd**2) - np.log(2*(np.pi)*xd*yd))

def gcn_image(image):
    image = image.astype(np.float)
    image -= image.reshape(-1,3).mean(axis=0)
    image /= image.reshape(-1,3).std(axis=0) + 1e-5
    return image

def write_skelton(image, co):
    l_weight = int(image.shape[0]/100.0)
    cv.line(image, co[0], co[1], color[0], l_weight)
    cv.line(image, co[1], co[2], color[1], l_weight)
    cv.line(image, co[3], co[4], color[2], l_weight)
    cv.line(image, co[4], co[5], color[3], l_weight)
    cv.line(image, co[10], co[11], color[4], l_weight)
    cv.line(image, co[11], co[12], color[5], l_weight)
    cv.line(image, co[13], co[14], color[6], l_weight)
    cv.line(image, co[14], co[15], color[7], l_weight)
    cv.line(image, co[6], co[7], color[8], l_weight)
    cv.line(image, co[7], co[8], color[9], l_weight)
    cv.line(image, co[8], co[9], color[10], l_weight)

    return image


if __name__ == '__main__' :
    args = get_arguments()
    
    #prepare model
    model = convolutional_pose_machine_REW.CPM_RW(args.n_point, args.n_stage)
    gpus = (args.gpu,)
    device = torch.device(f"cuda:{min(gpus)}" if len(gpus) > 0 else 'cpu')
    if args.unet_init_model:
        model.load_state_dict(torch.load(args.unet_init_model, map_location=device), strict=False)
    if args.cpm_init_model:
        model.load_state_dict(torch.load(args.cpm_init_model, map_location=device), strict=False)
    if args.vgg_init_model:
        model.load_state_dict(torch.load(args.vgg_init_model, map_location=device), strict=False)
    model.to(device)
    model.eval()
            
    #prepare directory
    mse=np.array([0]*16)
    #mae=np.array([0]*16)
    im_path = args.im_path
    paths = []
    unet_paths = []
    unet_spaths = []
    labels = []
    joint_xs = []
    joint_ys = []
    with open(args.csv,"r") as f:#dataset/mpii/.csv
        reader = csv.reader(f)
        for row in tqdm(reader):
            paths += [row[0]+'.jpg']
            
            unet_paths += ["{0}/{1}.jpg".format(im_path, row[0])]
            unet_spaths += ["{0}/{1}.png".format(args.seg_path, row[0])]
            labels += [[row[1]]+[row[2]]+[row[3]]+[row[4]]]#
            joint_xs += [list(map(float, row[5::2]))]
            joint_ys += [list(map(float, row[6::2]))]
        #m="{0:04d}".format(j)
        unet_test = ld.Loader(image_path=unet_paths, seg_path=unet_spaths)
        for i in tqdm(range(len(paths))):
            #print(paths[i])
            imagename=paths[i]
            resultname='r'+paths[i]
            label = labels[i]
            joint_x = joint_xs[i]
            joint_y = joint_ys[i]
            if not os.path.exists(os.path.join(im_path, imagename)):
                print('demo image does not exist')
                sys.exit()
            if not os.path.exists('demo_results'):
                os.mkdir('demo_results')

            #prepare image
            image = cv.imread(os.path.join(os.path.join(im_path, imagename)))
            w_image = cv.imread(os.path.join(os.path.join("dataset/mpii/13/org",imagename)))
            
            img = cv.resize(image, (224, 224))
            img = img[np.newaxis,:,:,:]
            img = torch.from_numpy(np.transpose(np.array(img).astype(np.float32), (0,3,1,2)))
            img = img.to(device)
            r_label = model.predict_vgg(img)
            add_label = r_label.cpu().detach().clone().numpy()
            add_label = np.squeeze(np.argmax(add_label,axis=1))
            #add_label = np.squeeze(np.argmax(label,axis=0))
            #print(add_label)
            
            #unet
            inputs = unet_test._data.images_original[i]
            #teacher = unet_._data.images_segmented[i]
            inputs = inputs[np.newaxis]
            #teacher = teacher[np.newaxis]
            inputs_torch = torch.from_numpy((np.transpose(np.array(inputs).astype(np.float32), (0,3,1,2))))
            inputs_torch = inputs_torch.to(device)
            #teacher_torch = torch.from_numpy((np.transpose(np.array(teacher).astype(np.float32), (0,3,1,2))))
            #teacher_torch = teacher_torch.to(device)
            
            h2 = model.predict_unet(inputs_torch)
            #unet_val_loss = loss.to('cpu').data
            h2 = h2.cpu().detach().clone().numpy()
            out = (np.transpose(np.array(h2).astype(np.float32), (0,2,3,1)))
            #out = np.squeeze(out[:,:,:,1])
            out = np.squeeze(np.argmax(out,axis=3))
            image[:,:,0]=0
            #print(label)
            if add_label == 0:
                image[:,:,1]=0
            elif add_label ==1:
                image[:,:,1]=50
            elif add_label ==2:
                image[:,:,1]=100
            elif add_label ==3:
                image[:,:,1]=150
            out = cv.cvtColor(cv.resize(out.astype(np.uint8),(400,800),interpolation=cv.INTER_CUBIC),cv.COLOR_GRAY2RGB)
            image = (image*(out*0.8))+(image*0.2)
            #-------------------------------------
            #cv.imwrite(os.path.join("images",imagename),image)
            #image = cv.imread(os.path.join("images", imagename))
            #-------------------------------------
            
            
            width = image.shape[1]
            height = image.shape[0]
            img = cv.resize(image, (args.im_size, args.im_size))
            img = gcn_image(img)
            c_map = makeGaussian(args.im_size, args.c_deviation*8, args.c_deviation*8, (args.im_size/2, args.im_size/2))
            img = img[np.newaxis,:,:,:]
            c_map = c_map[np.newaxis,np.newaxis,:,:]
            img = torch.from_numpy(np.transpose(np.array(img).astype(np.float32), (0,3,1,2)))
            img = img.to(device)
            c_map = torch.from_numpy(np.array(c_map).astype(np.float32))
            c_map = c_map.to(device)
            label = np.array(label)
            label = label.reshape((1,)+label.shape)
            label = label[np.newaxis]
            label = torch.from_numpy(np.array(label).astype(np.float32))#
            label = label.to(device)
            #label = addlabel(args.imagename)#
            #print(label)

            #start to demonstration
            h = model.predict_cpm(img, c_map, r_label)#
            #h = h.to(device)
            h2 = h.cpu().detach().clone().numpy()
            h3 = h2[0]

            #visualization
            b_maps = plt.figure(figsize=(10,10), dpi=1000)
            re_image = copy.copy(image)
            co_s = []
            for i in range(args.n_point):
                ax = b_maps.add_subplot(4,4,i+1)
                b_map = h3[i]
                #ax.imshow(image, extent=(0,width,0,height))
                #ax.imshow(b_map, extent=(0,width,0,height), alpha=0.5)
                #ax.set_title('${}$'.format(pose[i]))
                #ax.set_xticklabels([])
                #ax.set_yticklabels([])
                co = np.unravel_index(b_map.argmax(), b_map.shape)
                co_x = int(co[1]*width/46.0)
                co_y = int(co[0]*height/46.0)
                #cv.circle(w_image, (co_x, co_y), int(height/100), color[i], -1)
                co_s += [(co_x, co_y)]
                #print(joint_x[i],co_x)
                #print(joint_y[i],co_y)
                len=(((joint_x[i]-co_x)**2)+((joint_y[i]-co_y)**2))
                mse[i]+=len
                len2=math.sqrt(len)
                mae[i]+=len2
                    #print(point[i])
                #print(co_x,co_y)
            #re_image = write_skelton(w_image, co_s)
            #print(j)
            #save image
            #if args.resultname:
            #    result_imname = args.resultname
            #    result_bename = 'belief_maps_{}'.format(args.resultname)
            #else:
            #    result_imname = 'result_{}'.format(args.imagename)
            #    result_bename = 'belief_maps_{}'.format(args.imagename)
            #b_maps.savefig(os.path.join('demo_results',result_bename))
            #cv.imwrite(os.path.join('demo_results',result_imname), re_image)
            #print('completed')
        avgmse=0
        avgmae=0
        print('MSE')
        for n in range(0,16):
            #print(math.sqrt(mse[n]/140.0))
            avgmse+=mse[n]/140
        print('avgMSE')
        print(math.sqrt(avgmse/16))
        #print('MAE')
        #for n in range(0,16):
            #print(mae[n]/140.0)
        #    avgmae+=mae[n]/140
        #print('avgMAE')
        #print(avgmae/16)




