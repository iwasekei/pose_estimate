from skimage import transform
from tqdm import tqdm

import csv
import cv2 as cv
import numpy as np
import os
from PIL import Image

class Test(object):

    def __init__(self, raw_data, image_path, seg_path, imsize, pck_rate, c_deviation=2):
        for key, val in locals().items():
            setattr(self, key, val)
        self.load_data(self.raw_data, self.image_path, self.seg_path)

    def __len__(self):
        return len(self.images)

    def load_data(self, raw_data, image_path, seg_path):
        self.images = []
        #self.segs =[]
        self.joint_x = []
        self.joint_y = []
        self.bbox = []
        #self.cropsize = []
        self.im_path = []#
        #self.se_path = []
        self.unet_im_path = []
        self.unet_se_path =[]
        self.label = []#
        print ('preparing test dataset...')
        with open(raw_data,"r") as f:
            reader = csv.reader(f)
            for row in tqdm(reader):
                path = row[0]
                self.label += [[row[1]]+[row[2]]+[row[3]]+[row[4]]]#
                image = cv.imread("{0}/{1}.jpg".format(image_path, path))
                joint_x = np.array(list(map(float, row[5::2])),ndmin=1)
                joint_y = np.array(list(map(float, row[6::2])),ndmin=1)
                self.bbox += [[(max(joint_x,default=0) + min(joint_x,default=0)) / 2.0,
                        (max(joint_y,default=0) + min(joint_y,default=0)) / 2.0,
                        (max(joint_x,default=0) - min(joint_x,default=0)) * 1.2,
                        (max(joint_y,default=0) - min(joint_y,default=0)) * 1.2]]
                #image, seg, joint_x, joint_y, bbox = self.crop_image(image, seg, joint_x, joint_y, bbox)
                #self.cropsize += [[image.shape[1], image.shape[0]]]
                #image, seg, joint_x, joint_y = self.resize_image(image, seg, joint_x, joint_y)
                #image = self.gcn_image(image)
                self.im_path += [path]#
                self.unet_im_path += ["{0}/{1}.jpg".format(image_path, path)]
                self.unet_se_path += ["{0}/{1}.png".format(seg_path, path)]
                self.images += [image]
                self.joint_x += [joint_x]
                self.joint_y += [joint_y]

        print ('ready!')

    def generate(self, i, out, label):
        path = self.im_path[i]
        image = self.images[i]
        #NR, IE処理の追加
        #----------------------
        image[:,:,2]=0
        #print(label)
        if label == 0:
            image[:,:,1]=0
            labels = [1,0,0,0] 
        elif label ==1:
            image[:,:,1]=50
            labels = [0,1,0,0]
        elif label ==2:
            image[:,:,1]=100
            labels = [0,0,1,0]
        elif label ==3:
            image[:,:,1]=150
            labels = [0,0,0,1]
            
        out = Image.fromarray(np.uint8(out), mode='P')
        out = out.resize((400,800),Image.BICUBIC)
        out = out.convert('RGB')
        out = np.asarray(out)
        image = np.asarray(image)
        image = (image*(out*0.8))+(image*0.2)
        image = Image.fromarray(np.uint8(image))
        #image.save('1.jpg')
        image = np.asarray(image)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        #-----------------------
        
        #以下変更なし
        joint_x = np.array(self.joint_x[i])
        joint_y = np.array(self.joint_y[i])
        bbox = np.array(self.bbox[i]).copy()
        image, joint_x, joint_y, bbox = self.crop_image(image, joint_x, joint_y, bbox)
        image, joint_x, joint_y = self.resize_image(image, joint_x, joint_y)
        #self.joint_x[i] = joint_x
        #self.joint_y[i] = joint_y
        t = []
        h_map = np.zeros(int(self.imsize/8))
        for x, y in zip(joint_x, joint_y):
            b_map = self.makeGaussian(self.imsize/8, self.c_deviation, self.c_deviation, (x/8, y/8))
            h_map = h_map - b_map
            t += [b_map]
        t += [h_map]
        c_map = self.makeGaussian(self.imsize, self.c_deviation*8, self.c_deviation*8, (self.imsize/2, self.imsize/2))
        t = np.array(t)
        image = self.gcn_image(image)
        #label = np.array(labels)#
        #label = label.reshape((1,)+label.shape)
        return image, t, c_map
    
    def vgg_generate(self, i):
        path = self.im_path[i]
        image = self.images[i]
        image = cv.resize(image, (224, 224))
        label = np.array(self.label[i])#
        label = label.reshape((1,)+label.shape)
        return image, label#

    def evaluate(self, i, h):
        joint_x = self.joint_x[i]
        joint_y = self.joint_y[i]
        path = self.im_path[i]#
        image = self.images[i]
        bbox = np.array(self.bbox[i]).copy()
        image, joint_x, joint_y, bbox = self.crop_image(image, joint_x, joint_y, bbox)
        image, joint_x, joint_y = self.resize_image(image, joint_x, joint_y)
        is_correct = []
        for j in range(len(joint_x)):
            b_map = h[j]
            co = np.unravel_index(b_map.argmax(), b_map.shape)
            #print(co[0],co[1])
            #print(joint_y[j],joint_x[j])
            #print((joint_x[j] - co[1]*8)/float(self.imsize),(joint_y[j] - co[0]*8)/float(self.imsize),self.pck_rate)
            if pow((joint_x[j] - co[1]*8)/float(self.imsize),2) + pow((joint_y[j] - co[0]*8)/float(self.imsize),2) < pow(self.pck_rate,2):
                is_correct += [1]
            else:
                is_correct += [0]
        return is_correct

    def resize_image(self, image, joint_x, joint_y):
        joint_x = joint_x*self.imsize / image.shape[1]
        joint_y = joint_y*self.imsize / image.shape[0]
        image = cv.resize(image, dsize=(self.imsize, self.imsize))
        #seg = cv.resize(seg, dsize=(self.imsize, self.imsize))
        return image, joint_x, joint_y


    def crop_image(self, image, joint_x, joint_y, bbox):
        scale = 1.0
        bbox[2] = bbox[2]*scale
        bbox[3] = bbox[3]*scale
        bb_x1=int(bbox[0]-bbox[2]/2)
        bb_y1=int(bbox[1]-bbox[3]/2)
        bb_x2=int(bbox[0]+bbox[2]/2)
        bb_y2=int(bbox[1]+bbox[3]/2)
        if bb_x1<0 or bb_x2>image.shape[1] or bb_y1<0 or bb_y2 > image.shape[0]:
            pad = int(max(-(bb_x1), bb_x2-image.shape[1], -(bb_y1), bb_y2-image.shape[0]))
            image = np.pad(image, ((pad,pad),(pad,pad),(0,0)), 'constant')
            #seg = np.pad(seg, ((pad,pad),(pad,pad),(0,0)), 'constant')
        else:
            pad = 0
        image = image[bb_y1+pad:bb_y2+pad, bb_x1+pad:bb_x2+pad]
        #seg = seg[bb_y1+pad:bb_y2+pad, bb_x1+pad:bb_x2+pad]
        joint_x = (joint_x) - bb_x1
        joint_y = (joint_y) - bb_y1
        bbox[0] = bbox[0] - bb_x1
        bbox[1] = bbox[1] - bb_y1
        return image, joint_x, joint_y, bbox

    def gcn_image(self, image):
        image = image.astype(np.float)
        image -= image.reshape(-1,3).mean(axis=0)
        image /= image.reshape(-1,3).std(axis=0) + 1e-5
        return image


    def makeGaussian(self, size, xd, yd, center, order=1000):
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]
        x0 = center[0]
        y0 = center[1]
        return order*np.exp(-0.5*(((x-x0)**2)/xd**2 + ((y-y0)**2)/yd**2) - np.log(2*(np.pi)*xd*yd))
