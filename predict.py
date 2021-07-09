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



from os.path import join,  isfile #isdir, split, splitext, abspath

#from torchvision import datasets, transforms



from scipy.io import savemat
#%%


def predict(model_type, restore_path, save_dir, test_loader , output_ext, input_prefix=None , output_prefix=None):
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
    print("=> loaded checkpoint")
    

    # model
    model.eval()
    # setup the save directories
    dirs = ['side_1', 'side_2', 'side_3', 'side_4', 'side_5', 'fuse', 'merge', 'jpg_out']
    for idx, dir_path in enumerate(dirs):
        os.makedirs(join(save_dir, dir_path), exist_ok=True)
        if(idx < 6): os.makedirs(join(save_dir, 'mat' , dir_path), exist_ok=True)
    # run test
    #print(len(test_loader))
    for idx, batch in enumerate(test_loader):

        print("\rRunning test [%d/%d]" % (idx + 1, len(test_loader)), end='')
        data=batch['data']
        
        
        if (input_prefix is not None) and (output_prefix is not None):
            filename=batch['id'][0].replace(input_prefix, output_prefix)
        else:
            filename=batch['id'][0]
        

        if torch.cuda.is_available():
            for key in data:
                data[key] = data[key].cuda()
        _,_ , H, W = data['image'].shape
        with torch.no_grad():
            results = model(data)
        results_all = torch.zeros((len(results), 1, H, W))
       ## make our result array
        for i in range(len(results)):
          results_all[i, 0, :, :] = results[i]
        #filename = splitext(split(test_list[idx])[1])[0]
        torchvision.utils.save_image(results_all, join(save_dir, dirs[-1], f'{filename}.{output_ext}'))

        # now go through and save all the results
        for i, r in enumerate(results):

            img= torch.squeeze(r.detach()).cpu().numpy()
            savemat(join(save_dir,'mat',dirs[i],'{}.mat'.format(filename)), {'img': img})
            #img = Image.fromarray((img * 255).astype(np.uint8))

            #img.save(join(save_dir,dirs[i],'{}.jpg'.format(filename)))
            cv2.imwrite(join(save_dir,dirs[i],f'{filename}.{output_ext}'), (img * 255).astype(np.uint8))


        merge = sum(results) / 5
        torchvision.utils.save_image(torch.squeeze(merge), join(save_dir, dirs[-2], f'{filename}.{output_ext}'))
        torchvision.transforms.transforms.ToPILImage(torch.squeeze(results[i]))
    print('')

