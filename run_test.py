# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:07:05 2021

@author: yari
"""
from os.path import join,  isdir, dirname,  abspath  #split, splitext, isfile
from pathlib import Path

#from utils import Logger, arg_parser
from modules.data_loader import SnowData_s3, S3Images
 
#from modules.models import HED
from msnet import msNet

#from modules.options import arg_parser


from torch.utils.data import DataLoader
from datetime import datetime

from pandas import read_csv

from modules.predict import predict


AWS_ID='AKIAROLUNEJXA7YAGWXM' # <-- Turing # LUKE:'AKIAROLUNEJXGCGXYTNG'
AWS_KEY= 'T/04kyPqg1rQWYQ0TcfC2adyUwo3fLU5yCD6GXp8' # <-- Turing #Luke: 'N7kva2p99pVT+pDs14Vm4IiuG3GWyVO+O/9NiBv6'

#root=Path(__file__).parent.resolve()
root=Path(".")#macgregor', 'image')


BUCKET='cresis'
si=S3Images(AWS_ID, AWS_KEY)

# # train data load
# df_train=read_csv(root/'data/train.lst', header=None)
# images_2012=['greenland_picks_final_2009_2012_reformat/2012/'+item for item in df_train[0].values]

# #df_train2=read_csv(root/'data/sim2105.lst', header=None)
# #images_sim2105=['sim2105_fake_train/'+item for item in df_train2[0].values]

# ds_params={"bucket":BUCKET,  "s3Get": si.from_s3 , "keys": images_2012, 'wt':None}
# #ds_params2={"bucket":BUCKET,  "s3Get": si.from_s3 , "keys": images_sim2105, 'wt':None}

# ds=SnowData_s3(**ds_params)


# train_loader= DataLoader(ds, batch_size=1, shuffle=True)
#%%

# ds_params={"bucket":BUCKET,  "s3Get": si.from_s3 , 'wt':None}
# test_dataset=SnowData_s3(**ds_params, keys=images_test)
# test_loader= DataLoader(test_dataset, batch_size=1)
#     #test_root= Path('G:/My Drive/BinaLab/Datasets/Cresis/2012_main/test/image')

#%% 

BUCKET='binalab-data'
tag='ICE-210611-135614ms_ice12-MAIN'
df_fuse=read_csv(root/'data/test.lst', header=None)
images_fuse=[f'results/tmp/{tag}/final_test/fuse/'+item.replace('data','layer_binary') for item in df_fuse[0].values]

#%%

ds_params={"bucket":BUCKET,  "s3Get": si.from_s3 , 'wt':None}
fuse_dataset=SnowData_s3(**ds_params, keys=images_fuse)
fuse_loader= DataLoader(fuse_dataset, batch_size=1)
    #test_root= Path('G:/My Drive/BinaLab/Datasets/Cresis/2012_main/test/image')

#restore_path=Path(f'../tmp/{tag}/checkpoint_epoch{args.max_epoch}.pth')
max_epoch=15
restore_path=Path(f'../tmp/{tag}/checkpoint_epoch{max_epoch}.pth')
save_dir=Path(f'../{tag}/fuse2')

#%%
predict(model_type= msNet(),
    restore_path=restore_path,
    save_dir=save_dir,      
    test_loader=fuse_loader,
    output_ext='png'
    )

# predict(model_type= msNet(),
#     restore_path=restore_path,
#     save_dir=save_dir,      
#     test_loader=train_loader,
#     output_ext='png',
#     input_prefix='data',
#     output_prefix='layer_binary'
#     )