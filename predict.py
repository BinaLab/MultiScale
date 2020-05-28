# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:21:01 2020

@author: yari
"""


import os
import numpy as np
#from PIL import Image
import cv2
import torch

import torchvision

#from torch.utils.data import DataLoader
from torch.utils import data as D

from os.path import join, split, splitext, isdir, isfile, abspath

#from torchvision import datasets, transforms

from modules.data_loader import BasicDataset
from modules.models import HED
from torch.utils.data import DataLoader
#from modules.models_RCF import RCF

import glob
from scipy.io import savemat

from pathlib import Path
from datetime import datetime

#root=join('..','..','atasets','HED-BSDS')
#test_loader=get_dataLoaders_eval(root=root, image_dir=None , eval_image_list='test.lst' , ext='png' )
#%%

def main():
    tags=['ice2012-200527-082005HED_gitub2']
    for tag in tags:
        test_root= Path('G:/My Drive/BinaLab/Datasets/Cresis/2012_main/test/image')

    
        test_dataset=BasicDataset(test_root, ext='png')
        test_loader= DataLoader(test_dataset, batch_size=1)
        
        test(restore_path=Path(f'../tmp/{tag}/checkpoint_epoch15.pth'),
            save_dir=Path(f'C:/Users/yari/Documents/testsON12_trainedON12/{tag}'),
            model_type= HED(),
            test_loader=test_loader
            )
# =============================================================================


def test(restore_path, save_dir, model_type,test_loader):
    # model
    model = model_type
    #model = nn.DataParallel(model)
   # model.cuda()
    if torch.cuda.is_available():
        model.cuda()

    if isfile(restore_path):
        print("=> loading checkpoint '{}'".format(restore_path))
    else:
        raise('Restore path error!')

    if torch.cuda.is_available():
        checkpoint=torch.load(restore_path)
    else:
        checkpoint = torch.load(restore_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}'".format(restore_path))

    # model
    model.eval()
    # setup the save directories
    dirs = ['side_1', 'side_2', 'side_3', 'side_4', 'side_5', 'fuse', 'merge', 'jpg_out']
    for idx, dir_path in enumerate(dirs):
        os.makedirs(join(save_dir, dir_path), exist_ok=True)
        if(idx < 6): os.makedirs(join(save_dir, 'mat' , dir_path), exist_ok=True)
    # run test
    #print(len(test_loader))
    for idx, data in enumerate(test_loader):

        print("\rRunning test [%d/%d]" % (idx + 1, len(test_loader)), end='')
        image=data['image']

        filename=data['id'][0].replace('image', 'layer_binary')
        

        if torch.cuda.is_available():
            image = image.cuda()
        _,_ , H, W = image.shape
        results = model(image)
        results_all = torch.zeros((len(results), 1, H, W))
       ## make our result array
        for i in range(len(results)):
          results_all[i, 0, :, :] = results[i]
        #filename = splitext(split(test_list[idx])[1])[0]
        torchvision.utils.save_image(results_all, join(save_dir, dirs[-1], '{}.jpg'.format(filename)))

        # now go through and save all the results
        for i, r in enumerate(results):

            img= torch.squeeze(r.detach()).cpu().numpy()
            savemat(join(save_dir,'mat',dirs[i],'{}.mat'.format(filename)), {'img': img})
            #img = Image.fromarray((img * 255).astype(np.uint8))

            #img.save(join(save_dir,dirs[i],'{}.jpg'.format(filename)))
            cv2.imwrite(join(save_dir,dirs[i],'{}.jpg'.format(filename)), (img * 255).astype(np.uint8))


        merge = sum(results) / 5
        torchvision.utils.save_image(torch.squeeze(merge), join(save_dir, dirs[-2], '{}.jpg'.format(filename)))
        torchvision.transforms.transforms.ToPILImage(torch.squeeze(results[i]))
    print('')

# # # # # # # # # # # # # #
if __name__ == '__main__':
    main()
