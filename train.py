#!/user/bin/python
# coding=utf-8

#from utils import Logger, arg_parser


import os
from os.path import join, isdir
from pathlib import Path

#from utils import Logger, arg_parser

from msnet import msNet
from modules.data_loader import SnowData
from modules.trainer import Trainer, Network 
from modules.utils import struct
from modules.transforms import Fliplr, Rescale_byrate
#from modules.options import arg_parser


from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime





root=Path("../cresis-data/")

tag = datetime.now().strftime("%y%m%d-%H%M%S")


params={
     'root': root,
     'trainlist':Path('./data/train.lst'),
     'devlist':Path('./data/dev.lst'),
     'tmp': Path(f'../tmp/{tag}'), ##os.getcwd()
     'log_dir': Path(f'../logs/{tag}'),
     'val_percent': 0,
     'start_epoch' : 0,
     'max_epoch' : 15,
     'batch_size': 1,
     'itersize': 10,
     'stepsize': 1e4,
     'lr': 1e-06,
     'momentum': 0.9,
     'weight_decay': 0.0002,
     'gamma': 0.1,
     'pretrained_path': None,
     'resume_path': None,
     'weights_init_on': False,
     'multi_gpu': True
     }

args= struct(**params)


def main():

    if not isdir(args.tmp):
        os.makedirs(args.tmp)

    # define network
    net=Network(args, model=msNet())

    ds=[SnowData(root=root,lst=args.trainlist),
    SnowData(root=root,lst=args.trainlist, transform=Rescale_byrate(.75)),
    SnowData(root=root,lst=args.trainlist,transform=Rescale_byrate(.5)),
    SnowData(root=root,lst=args.trainlist,transform=Rescale_byrate(.25)),
    SnowData(root=root,lst=args.trainlist, transform=Fliplr())
    ]
    #train_dataset=SnowData(root=root,lst=args.trainlist)
    train_dataset=ConcatDataset(ds)
    train_loader= DataLoader(train_dataset, num_workers=8, batch_size=1, shuffle=True)

    # development dataset (optional):
    if args.devlist is not None:
        dev_dataset=SnowData(root=root,lst=args.devlist)
        dev_loader= DataLoader(dev_dataset, batch_size=1)

    # define trainer
    trainer=Trainer(args,net, train_loader=train_loader)

    # switch to train mode: not needed!  model.train()
    for epoch in range(args.start_epoch, args.max_epoch):
        ## initial log (optional:sample36)
        if (epoch == 0) and (args.devlist is not None):
            print("Performing initial testing...")
            trainer.dev(dev_loader=dev_loader,save_dir = join(args.tmp, 'testing-record-0-initial'), epoch=epoch)
    
        ## training
        trainer.train(save_dir = args.tmp, epoch=epoch)
    
        ## dev check (optional:sample36)
        if args.devlist is not None:
            trainer.dev(dev_loader=dev_loader,save_dir = join(args.tmp, f'testing-record-epoch-{epoch+1}'), epoch=epoch)
        
if __name__ == '__main__':
    main()
