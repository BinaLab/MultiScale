# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 10:07:05 2021

@author: yari
"""
from os.path import join,  isdir, dirname,  abspath  #split, splitext, isfile
from pathlib import Path

from modules.data_loader import SnowData

from msnet import msNet

from torch.utils.data import DataLoader

from modules.predict import predict



#root=Path(__file__).parent.resolve()
root=Path(".")

tag='last_experiment'
#%%
test_dataset=SnowData(root=root,lst='.\data\test.lst')
test_loader= DataLoader(test_dataset, batch_size=1)
    #test_root= Path('G:/My Drive/BinaLab/Datasets/Cresis/2012_main/test/image')

#restore_path=Path(f'../tmp/{tag}/checkpoint_epoch{args.max_epoch}.pth')
max_epoch=15
restore_path=Path(f'../tmp/{tag}/checkpoint_epoch{max_epoch}.pth')
save_dir=Path(f'../{tag}/test_result')

#%%
predict(model_type= msNet(),
    restore_path=restore_path,
    save_dir=save_dir,      
    test_loader=test_loader,
    output_ext='png'
    )

# predict(model_type= msNet(),
#     restore_path=restore_path,
#     save_dir=save_dir,      
#     test_loader=test_loader,
#     output_ext='png',
#     input_prefix='data',
#     output_prefix='layer_binary'
#     )