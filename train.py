#!/user/bin/python
# coding=utf-8

#from utils import Logger, arg_parser


import os
from os.path import join, isdir
from pathlib import Path

#from utils import Logger, arg_parser


from msnet import msNet
from modules.data_loader import SnowData_s3, S3Images
from modules.trainer import Trainer, Network 
from modules.utils import struct
from modules.transforms import Fliplr, Rescale_byrate
#from modules.options import arg_parser


from torch.utils.data import DataLoader, ConcatDataset
from datetime import datetime
from torch import save, cuda, device

from pandas import read_csv
import logging



AWS_ID=""
AWS_KEY= ""



tag = datetime.now().strftime("%y%m%d-%H%M%S")


TMP='tmp'
LOGS='logs'
LOGGER='logger'


params={
     'root': Path("."),
     'trainlist':Path('./data/train.lst'),
     'devlist':None, #Path('./data/dev.lst'),
     'tmp_dir': None, # Path(f'../{TMP}/{tag}'), ##os.getcwd()
     'log_dir': None, #Path(f'../{LOGS}/{tag}'),
     'logger_dir': Path(f'../{LOGGER}'),
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
     'final' : join('..',tag)
     }

args= struct(**params)


#%%logger
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



#%% train data loader
t0=datetime.now()
BUCKET='cresis'
si=S3Images(AWS_ID, AWS_KEY)

# train data load
df_train=read_csv(args.root/'data/train.lst', header=None)
images_2012=['greenland_picks_final_2009_2012_reformat/2012/'+item for item in df_train[0].values]

#df_train2=read_csv(args.root/'data/sim2105.lst', header=None)
#images_sim2105=['sim2105_fake_train/'+item for item in df_train2[0].values]

ds_params={"bucket":BUCKET,  "s3Get": si.from_s3 , "keys": images_2012, 'wt':None}
#ds_params2={"bucket":BUCKET,  "s3Get": si.from_s3 , "keys": images_sim2105, 'wt':None}

ds=[
SnowData_s3(**ds_params),
SnowData_s3(**ds_params, transform=Rescale_byrate(.75)),
SnowData_s3(**ds_params,transform=Rescale_byrate(.5)),
SnowData_s3(**ds_params,transform=Rescale_byrate(.25)),
SnowData_s3(**ds_params, transform=Fliplr())
# ,SnowData_s3(**ds_params2),
# SnowData_s3(**ds_params2, transform=Rescale_byrate(.75)),
# SnowData_s3(**ds_params2,transform=Rescale_byrate(.5)),
# SnowData_s3(**ds_params2,transform=Rescale_byrate(.25)),
# SnowData_s3(**ds_params2, transform=Fliplr())
]

train_dataset=ConcatDataset(ds)
train_loader= DataLoader(train_dataset, num_workers=8, batch_size=1, shuffle=True)



if args.logger is not None:
    dt=datetime.now()-t0
    args.logger.info(f"Loading {len(train_dataset)} images in  {dt.total_seconds()} seconds")

# development dataset (optional):
if args.devlist is not None:
    # develpment data load
    df_dev=read_csv(args.root/'data/dev.lst', header=None)
    images_dev=['greenland_picks_final_2009_2012_reformat/2012/'+item for item in df_dev[0].values]
    
    dev_dataset=SnowData_s3(bucket='cresis', keys=images_dev, s3Get=si.from_s3)
    dev_loader= DataLoader(dev_dataset, batch_size=1)
            
#%% define network and trainer
net=Network(args, model=msNet())

# define trainer
trainer=Trainer(args,net, train_loader=train_loader)

#%% Training
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

if args.tmp_dir is None:
    if not isdir(args.final):
        os.makedirs(args.final)   
    t0=datetime.now()       
    save(trainer.final_state, join(args.final,f'final_{args.max_epoch}.pth'))
    
    dt=datetime.now()-t0
    args.logger.info(f"Time to save final state : {dt.total_seconds()} seconds")
    
#%% Test results




from modules.predict import predict

df_test=read_csv(args.root/'data/test.lst', header=None)
images_test=['greenland_picks_final_2009_2012_reformat/2012/'+item for item in df_test[0].values]


ds_params={"bucket":BUCKET,  "s3Get": si.from_s3 , 'train' : False, 'wt':None}
test_dataset=SnowData_s3(**ds_params, keys=images_test)
test_loader= DataLoader(test_dataset, batch_size=1)

if args.tmp_dir is  None:
    restore_path=join(args.final,f'final_{args.max_epoch}.pth')
    save_dir=join(args.final, 'final_test')
else:
    restore_path=Path(f'../{args.tmp}/{tag}/checkpoint_epoch{args.max_epoch}.pth')
    save_dir=Path(f'../{args.tmp}/{tag}/final_test')


predict(model_type= msNet(),
    restore_path=restore_path,
    save_dir=save_dir,      
    test_loader=test_loader,
    output_ext='png',
    input_prefix='data',
    output_prefix='layer_binary'
    )