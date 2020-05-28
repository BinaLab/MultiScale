# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:26:56 2020

@author: yari
"""

from torch.utils import data as D
from os.path import join, abspath #,split, abspath, splitext, split, isdir, isfile
import numpy as np
import cv2
import os

import pandas as pd

import glob

from scipy.io import loadmat



class Dataset_ls(D.Dataset):
    """
    dataset from list
    returns data after preperation
    """
    def __init__(self, root, lst, transform=None):
        self.df=pd.read_csv(join(root, lst), delimiter=' ', names=['data', 'ctour'])
        self.root=os.path.abspath(root)
        self.transform=transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # get image (jpg)
        img_abspath= join(self.root, self.df['data'][index])
        assert os.path.isfile(img_abspath), "file  {}. doesn't exist.".format(img_abspath)

               # Edge Maps (binary files)
        ct_abspath=join(self.root, self.df['ctour'][index])
        assert os.path.isfile(ct_abspath), "file  {}. doesn't exist.".format(ct_abspath)

        img=cv2.imread(img_abspath)
        ctour=cv2.imread(ct_abspath, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            img=self.transform(img)
            ctour=self.transform(ctour)

        img, ctour= prepare_img_3c(img), prepare_ctour(ctour)

        (data_id, _) = os.path.splitext(os.path.basename(img_abspath))

        return {'image': img, 'mask' : ctour , 'id': data_id}



class BasicDataset(D.Dataset):
    """
    dataset from directory
    returns img after preperation, no label
    """
    def __init__(self, root, ext):
        self.root=root
        self.ext=ext
        self.rel_paths=glob.glob(join(root, '*.{}'.format(ext)))

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, index):

        # get image
        img_abspath= abspath(self.rel_paths[index])
        assert os.path.isfile(img_abspath), "file  {}. doesn't exist.".format(img_abspath)

        img=cv2.imread(img_abspath)

        img= prepare_img_3c(img)

        (data_id, _) = os.path.splitext(os.path.basename(img_abspath))


        return {'image': img, 'id':data_id}


class dataset_mat(D.Dataset):
    """
    dataset from list
    """
    def __init__(self, root, transform=None):
         self.root=root
         self.rel_paths=glob.glob(join(root, '*.mat'))
         self.transform=transform
         self.len= len(self.rel_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        # get mat file
        abs_path= abspath(self.rel_paths[index])
        assert os.path.isfile(abs_path), "file  {}. doesn't exist.".format(abs_path)

        data=loadmat(abs_path)
        img=data['data']
        ctour=data['raster']


        if self.transform:
            img=self.transform(img)
            ctour=self.transform_lb(ctour)

        img, ctour= prepare_img_mat(img), prepare_ctour(ctour)

        #(data_id, _) = os.path.splitext(os.path.basename(abs_path))
        data_id=data['fn_name'][0]


        return {'image': img, 'mask' : ctour , 'id': data_id}

def prepare_img_mat(img):
        img=np.array(img, dtype=np.float32)
        img=img*255/np.max(img)
        img=np.expand_dims(img, axis=2)
        img=np.repeat(img,3,axis=2)
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img=img.transpose(2,0,1)
        return img

def prepare_ctour(ctour):
        #ctour=np.array(ctour, dtype=np.float32)
        ctour = (ctour > 0 ).astype(np.float32)
        ctour=np.expand_dims(ctour,axis=0)
        return ctour

def prepare_img_3c(img):
        img=np.array(img, dtype=np.float32)
        img -= np.array((104.00698793,116.66876762,122.67891434))
        img=img.transpose(2,0,1)
        return img




# enum ImreadModes
# {
#     IMREAD_UNCHANGED           = -1,
#     IMREAD_GRAYSCALE           = 0,
#     IMREAD_COLOR               = 1,
#     IMREAD_ANYDEPTH            = 2,
#     IMREAD_ANYCOLOR            = 4,
#     IMREAD_LOAD_GDAL           = 8,
#     IMREAD_REDUCED_GRAYSCALE_2 = 16,
#     IMREAD_REDUCED_COLOR_2     = 17,
#     IMREAD_REDUCED_GRAYSCALE_4 = 32,
#     IMREAD_REDUCED_COLOR_4     = 33,
#     IMREAD_REDUCED_GRAYSCALE_8 = 64,
#     IMREAD_REDUCED_COLOR_8     = 65,
#     IMREAD_IGNORE_ORIENTATION  = 128,}
