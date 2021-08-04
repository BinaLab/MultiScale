# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:26:56 2020

@author: yari
"""

from torch.utils import data as D
from os.path import join, splitext, basename #,split, abspath, splitext, split, isdir, isfile
import numpy as np
import cv2
import os

import pandas as pd

#S3
import boto3
from PIL import Image
from io import BytesIO


class SnowData(D.Dataset):
    """
    dataset from list
    returns data after preperation
    """
    def __init__(self, root, lst, train=True, transform=None,  wt =None):
        self.df=pd.read_csv(lst, names=['data'])
        self.root=root #os.path.abspath(root)
        self.transform=transform
        self.train=train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # get image (jpg)
        img_abspath= join(self.root, self.df['data'][index])
        assert os.path.isfile(img_abspath), "file  {}. doesn't exist.".format(img_abspath)

               # Edge Maps (binary files)


        img=cv2.imread(img_abspath,0)
        if self.transform:
            img=self.transform(img)
                
            #### will be added later
        # if self.wt is not None:
        #     data=get_wt(img, self.wt , mode='periodic', level=4)
        # else:
        #     data={}
        data={}
        img = prepare_img(img)
        data['image']=img
        
         #img=img[0,:,:]*np.ones(1, dtype=np.float32)[None, None, :]
        (data_id, _) = splitext(basename(img_abspath))
        if self.train:
            ct_abspath=img_abspath.replace('data_','layer_binary_')
            assert os.path.isfile(ct_abspath), "file  {}. doesn't exist.".format(ct_abspath)
            ctour=cv2.imread(ct_abspath, cv2.IMREAD_GRAYSCALE)
            if self.transform:
                ctour=self.transform(ctour)
            ctour= prepare_ctour(ctour)
           
    
            return {'data': data, 'label': ctour, 'id': data_id}
        else:    
            return {'data': data,  'id': data_id}            


class SnowData_s3(D.Dataset):
    """
    dataset from list
    returns data after preperation
    """
    def __init__(self, bucket, keys, s3Get,  wt =None, transform=None):
        self.df=keys
        self.bucket=bucket
        self.s3Get=s3Get
        self.transform=transform
        #self.wt=wt
       # self.prepare=prepare

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        # get image (jpg)
        key=self.df[index]
        img= self.s3Get(self.bucket, key)


        if self.transform:
            img=self.transform(img)
                
            #### will be added later
        # if self.wt is not None:
        #     data=get_wt(img, self.wt , mode='periodic', level=4)
        # else:
        #     data={}
        data={}
        img = prepare_img(img)
        data['image']=img
        
         #img=img[0,:,:]*np.ones(1, dtype=np.float32)[None, None, :]
        (data_id, _) = os.path.splitext(os.path.basename(key))
        
        if self.train:
            ctour=self.s3Get(self.bucket, key.replace('data', 'layer_binary'))
            if self.transform:
                ctour=self.transform(ctour)
            ctour= prepare_ctour(ctour)
           
    
            return {'data': data, 'label': ctour, 'id': data_id}
        else:    
            return {'data': data,  'id': data_id}   


        
def prepare_img(img):
        img=np.array(img, dtype=np.float32)
        #img=np.expand_dims(img,axis=2)
        (R,G,B)=(104.00698793,116.66876762,122.67891434)
        img -= np.array((0.299*R + 0.587*G + 0.114*B))
        #img=img*np.ones(1, dtype=np.float32)[None, None, :]
        #img=img.transpose(2,0,1)
        return np.expand_dims(img,axis=0)    


def prepare_ctour(ctour):
        #ctour=np.array(ctour, dtype=np.float32)
        ctour = (ctour > 0 ).astype(np.float32)
        return np.expand_dims(ctour,axis=0)



def prepare_w(img):
        img=np.array(img, dtype=np.float32)
        img=np.expand_dims(img,axis=0)
        return img



class S3ImagesUploadFailed(Exception):
    pass

class S3Images(object):

    """Useage:

        images = S3Images(aws_access_key_id='fjrn4uun-my-access-key-589gnmrn90',
                          aws_secret_access_key='4f4nvu5tvnd-my-secret-access-key-rjfjnubu34un4tu4',
                          region_name='eu-west-1')
        im = images.from_s3('my-example-bucket-9933668', 'pythonlogo.png')
        im
        images.to_s3(im, 'my-example-bucket-9933668', 'pythonlogo2.png')
    """

    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name=None):
        self.s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                                     aws_secret_access_key=aws_secret_access_key,
                                     region_name=region_name)


    def from_s3(self, bucket, key):
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        return Image.open(obj["Body"])


    def to_s3(self, img, bucket, key):
        buffer = BytesIO()
        img.save(buffer, self.__get_safe_ext(key))
        buffer.seek(0)
        sent_data = self.s3.put_object(Bucket=bucket, Key=key, Body=buffer)
        if sent_data['ResponseMetadata']['HTTPStatusCode'] != 200:
            raise S3ImagesUploadFailed('Failed to upload image {} to bucket {}'.format(key, bucket))

# def wt_scale(wt):
#     return 255*(wt-np.min(wt))/(np.max(wt)-np.min(wt))

# def get_wt(im, wname, mode, level,scaleit=False):
#     w=pywt.wavedec2(im,wname, mode=mode, level=level)
#     if scaleit:
#         wt={f'cA{level}': prepare_w(wt_scale(w[0]))}
#         for i in range(1,level):
#             wt.update({f'cH{i}': prepare_w(wt_scale(w[-i][0]))})
#             wt.update({f'cV{i}': prepare_w(wt_scale(w[-i][1]))})
#             wt.update({f'cD{i}': prepare_w(wt_scale(w[-i][2]))})
#     else:
#         wt={f'cA{level}': prepare_w(w[0])}
#         for i in range(1,level):
#             wt.update({f'cH{i}': prepare_w(w[-i][0])})
#             wt.update({f'cV{i}': prepare_w(w[-i][1])})
#             wt.update({f'cD{i}': prepare_w(w[-i][2])})
#     return wt



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
