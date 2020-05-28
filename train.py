#!/user/bin/python
# coding=utf-8
import os
from os.path import join,  isdir, dirname,  abspath  #split, splitext, isfile
from pathlib import Path

#from utils import Logger, arg_parser
from modules.data_loader import Dataset_ls, BasicDataset
#from modules.models import HED
from modules.models_RCF import RCF
from modules.trainer import Network, Trainer
from modules.utils import struct
#from modules.options import arg_parser

import torch.cuda
from torch.utils.data import DataLoader
from datetime import datetime




root=Path("G:/My Drive/BinaLab/Datasets/Cresis/2012_main/")#macgregor', 'image')
tag = datetime.now().strftime("%y%m%d-%H%M%S")+'RCF_gitub'


params={
     'root': root,
     'tmp': Path(f'../tmp/ice2012-{tag}'), ##os.getcwd()
     'log_dir': Path(f'../logs/ice2012-{tag}'),
     'dev_dir':root/'sample36',
     'val_percent': 0,
     'start_epoch' : 0,
     'max_epoch' : 15,
     'batch_size': 1,
     'itersize': 10,
     'stepsize': 3,
     'lr': 1e-06,
     'momentum': 0.9,
     'weight_decay': 0.0002,
     'gamma': 0.1,
     'pretrained_path': None,
     'resume_path': None,
     'use_cuda': torch.cuda.is_available()
     }

args= struct(**params)


def main():

    if not isdir(args.tmp):
        os.makedirs(args.tmp)

    # define network
    net=Network(args, model=RCF())

    # train dataset
    train_dataset=Dataset_ls(root=root,lst='train_pair.lst')
    train_loader= DataLoader(train_dataset, batch_size=1, shuffle=True)

    # development dataset (optional)
    dev_dataset=BasicDataset(args.dev_dir, ext='png')
    dev_loader= DataLoader(dev_dataset, batch_size=1)

    # define trainer
    trainer=Trainer(args,net, train_loader=train_loader)

    # switch to train mode: not needed!  model.train()
    for epoch in range(args.start_epoch, args.max_epoch):

        ## initial log (optional:sample36)
        if epoch == 0:
            print("Performing initial testing...")
            trainer.dev(dev_loader=dev_loader,save_dir = join(args.tmp, 'testing-record-0-initial'))

        ## training
        trainer.train(save_dir = args.tmp, epoch=epoch)

        ## dev check (optional:sample36)
        trainer.dev(dev_loader=dev_loader,save_dir = join(args.tmp, 'testing-record-epoch-%d' % epoch))

if __name__ == '__main__':
    main()
