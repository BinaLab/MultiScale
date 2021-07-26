#!/user/bin/python
# coding=utf-8

#from utils import Logger, arg_parser


import os
from os.path import join, isdir
from pathlib import Path

#from utils import Logger, arg_parser

#from modules.models import HED
from msnet import msNet
from modules.data_loader import SnowData
from modules.trainer import Trainer, Network 
from modules.utils import struct
from modules.transforms import Fliplr, Rescale_byrate
#from modules.options import arg_parser


from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime
from torch import save, cuda, device

import logging



root=Path("../cresis-data/")

tag = datetime.now().strftime("%y%m%d-%H%M%S")


params={
     'root': root,
     'trainlist':Path('./data/train.lst'),
     'devlist':None, #Path('./data/dev.lst'),
     'tmp_dir': None, # Path(f'../tmp/{tag}'), ##os.getcwd()
     'log_dir': None, #Path(f'../logs/{tag}'),
     'logger_dir': Path('../logger'),
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
     'multi_gpu': True,
     'device' : device("cuda:0" if cuda.is_available() else "cpu"),
     }

args= struct(**params)





#%%
def main():
    
    ## logger
    #LOG_FORMAT="%(Levelname)s %(asctime)s - %(message)s"
    if args.logger_dir is not None:
        if not isdir(args.logger_dir):
                os.makedirs(args.logger_dir)
        
        LOG_FORMAT = '%(message)s'
        logging.basicConfig(filename=args.logger_dir/f"log_{tag}.log",
                            level=logging.DEBUG, 
                            format=LOG_FORMAT)
                            #,filemode='w')
        
        args.logger=logging.getLogger()
    else:
        args.logger=None

    if (args.tmp_dir is not None) and (not isdir(args.tmp_dir)):
        os.makedirs(args.tmp_dir)

    # define network
    net=Network(args, model=msNet())

    t0=datetime.now()
    
    ds=[SnowData(root=root,lst=args.trainlist),
    SnowData(root=root,lst=args.trainlist, transform=Rescale_byrate(.75)),
    SnowData(root=root,lst=args.trainlist,transform=Rescale_byrate(.5)),
    SnowData(root=root,lst=args.trainlist,transform=Rescale_byrate(.25)),
    SnowData(root=root,lst=args.trainlist, transform=Fliplr())
    ]
    #train_dataset=SnowData(root=root,lst=args.trainlist)
    train_dataset=ConcatDataset(ds)
    train_loader= DataLoader(train_dataset, num_workers=8, batch_size=1, shuffle=True)
    
    
    
    if args.logger is not None:
        dt=datetime.now()-t0
        args.logger.info(f"Loading {len(train_dataset)} images in  {dt.total_seconds()} seconds")

    # development dataset (optional):
    if args.devlist is not None:
        dev_dataset=SnowData(root=root,lst=args.devlist)
        dev_loader= DataLoader(dev_dataset, batch_size=1)

    # define trainer
    trainer=Trainer(args,net, train_loader=train_loader)

    # switch to train mode: not needed!  model.train()
    t1=datetime.now()
    for epoch in range(args.start_epoch, args.max_epoch):
        ## initial log (optional:sample36)
        if (epoch == 0) and (args.tmp_dir is not None) and (args.devlist is not None):
            print("Performing initial testing...")
            trainer.dev(dev_loader=dev_loader,save_dir = join(args.tmp_dir, 'testing-record-0-initial'), epoch=epoch)
    
        ## training
        t0=datetime.now()
        trainer.train(epoch=epoch)
        if args.logger is not None:
            dt=datetime.now()-t0
            args.logger.info(f"Training time for epoch {epoch}: {dt.total_seconds()} seconds")
    
        ## dev check (optional:sample36)
        if (args.tmp_dir is not None) and args.devlist is not None:
            trainer.dev(dev_loader=dev_loader,save_dir = join(args.tmp_dir, f'testing-record-epoch-{epoch+1}'), epoch=epoch)
        
    dt=datetime.now()-t1
    args.logger.info(f"Total training time : {dt.total_seconds()} seconds")
    
    final=join('..',tag)
    if not isdir(final):
        os.makedirs(final)   
    t0=datetime.now()       
    save(trainer.final_state, join(final,f'final_{args.max_epoch}.pth'))
    
    dt=datetime.now()-t0
    args.logger.info(f"Time to save final state : {dt.total_seconds()} seconds")
    
        
if __name__ == '__main__':
    main()