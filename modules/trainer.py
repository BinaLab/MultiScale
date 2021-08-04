import os
import numpy as np
#from PIL import Image
#import time
import torch

#import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
#from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from torch.optim import lr_scheduler
#import torchvision
import cv2


from os.path import join, split, isdir, isfile, splitext

#from modules.models_aux import  weights_init #, convert_vgg 
from modules.functions import   cross_entropy_loss # sigmoid_cross_entropy_loss
from modules.utils import Averagvalue #, save_checkpoint



#from tensorboardX import SummaryWriter



class Network(object):
    def __init__(self, args, model):
        super(Network, self).__init__()
        # a necessary class for initialization and pretraining, there are precision issues when import model directly

        model.to(args.device)
        
        if args.multi_gpu:
            self.model = nn.DataParallel(model)
        else:
            self.model = model
            
        if args.weights_init_on:
            self.model.apply(weights_init)


        # if args.pretrained_path is not None:
        #     self.model.apply(weights_init)
        #     vgg_pretrain(model=model, pretrained_path=args.pretrained_path)

        if args.resume_path is not None:
            resume(model=model, resume_path=args.resume_path)

        # if torch.cuda.is_available():
        #     self.model.cuda()


class Trainer(object):
    def __init__(self, args, net, train_loader, val_loader=None):
        super(Trainer, self).__init__()

        self.model=net.model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.n_train=len(train_loader)
        if val_loader is not None:
            self.n_val=len(val_loader)
        else:
            self.n_val=0

        self.n_dataset= self.n_train+self.n_val
        self.global_step = 0

        self.batch_size=args.batch_size

        #losses
        self.train_loss = []
        self.train_loss_detail = []

        self.val_loss = []
        self.val_loss_detail = []

        self.max_epoch=args.max_epoch
    

        self.use_cuda=torch.cuda.is_available()

        # training args
        self.itersize=args.itersize

        #tune lr
        tuned_lrs=tune_lrs(self.model,args.lr, args.weight_decay)

        self.optimizer = torch.optim.SGD(tuned_lrs, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=args.stepsize, gamma=args.gamma)
        
        self.final_state= None
        
        if args.log_dir is not None:
            self.writer = SummaryWriter(args.log_dir)
        else:
            self.writer =None
            
        
        
        #self.logger= args.logger

            
    def train(self, epoch, tmp_dir= None):

        ## initilization
            
        losses = Averagvalue()
        epoch_loss = []

        val_losses = Averagvalue()
        epoch_val_loss = []


        counter = 0
        with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{self.max_epoch}', unit='img') as pbar:
            for batch in self.train_loader:

                data, label, image_name= batch['data'], batch['label'], batch['id'][0]
                
                

                if torch.cuda.is_available():
                    for key in data:
                        data[key]=data[key].cuda()
                    label=label.cuda()
                    
                image=data['image']
                
                ## forward
                outputs = self.model(data)
                ## loss


                if self.use_cuda:
                    loss = torch.zeros(1).cuda()
                else:
                    loss = torch.zeros(1)


                for o in outputs:
                    loss = loss+cross_entropy_loss(o, label)
                #loss=self.loss_w(loss_r)

                counter += 1
                loss = loss / self.itersize
                loss.backward()

                # SDG step
                if counter == self.itersize:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    counter = 0
                    #adjust learnig rate
                    self.scheduler.step()
                    self.global_step += 1

                # measure accuracy and record loss
                losses.update(loss.item(), image.size(0))
                epoch_loss.append(loss.item())

                
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(image.shape[0])

                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
    
                if (self.global_step >0) and (self.global_step % 500 ==0): #(self.n_dataset // (10 * self.batch_size)) == 0:
        
                    if self.writer is not None:
                        #tensorboard
                        for tag, value in self.model.named_parameters():
                            tag = tag.replace('.', '/')
                            self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), self.global_step)
                            self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), self.global_step)
    
    
                        self.writer.add_images('images', image, self.global_step)
                        self.writer.add_images('masks/true', label, self.global_step)
                        self.writer.add_images('masks/pred', outputs[-1] > 0.5, self.global_step)


                    if tmp_dir is not None:
                        outputs.append(label)
                        outputs.append(image)
                        dev_checkpoint(save_dir=join(tmp_dir, f'training-epoch-{epoch+1}-record'),
                                   i=self.global_step, epoch=epoch, image_name=image_name, outputs= outputs)

        if tmp_dir is not None:
            self.save_state(epoch, save_path=join(tmp_dir , f'checkpoint_epoch{epoch+1}.pth'))
        if epoch==self.max_epoch-1:
            self.final_state= {'epoch': epoch,
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict()
                                }
        if self.writer is not None:
            self.writer.add_scalar('Loss_avg', losses.avg, epoch+1)
            if val_losses.count>0:
                self.writer.add_scalar('Val_Loss_avg', val_losses.avg, epoch+1)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
        self.train_loss.append(losses.avg)
        self.train_loss_detail += epoch_loss

    def dev(self,dev_loader, save_dir, epoch):
        print("Running test ========= >")
        self.model.eval()
        if not isdir(save_dir):
            os.makedirs(save_dir)
        for idx, batch in enumerate(dev_loader):

            #image,image_id= batch['image'] ,  batch['id'][0]
            
            data, label, image_name= batch['data'], batch['label'], batch['id'][0]

            _, _, H, W = data['image'].shape
            
            if torch.cuda.is_available():
                for key in data:
                    data[key]=data[key].cuda()

            

            with torch.no_grad():
               outputs = self.model(data)

            outputs.append(1-outputs[-1])
            outputs.append(label)
            dev_checkpoint(save_dir, -1, epoch, image_name, outputs)
            
            # result=tensor2image(results[-1])
            # result_b=tensor2image(1-results[-1])

            # cv2.imwrite(join(save_dir, f"{image_id}.png".replace('image','fuse')), result)
            # cv2.imwrite(join(save_dir, f"{image_id}.jpg".replace('image','fuse')), result_b)


    def save_state(self, epoch, save_path='checkpoint.pth'):
        torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, save_path)

##========================== initial state

def weights_init(m):
    """ Weight initialization function. """
    if isinstance(m, nn.Conv2d):
        # Initialize: m.weight.
        # xavier(m.weight.data) #init 1
        #m.weight.data.normal_(0, 0.01) #init 2
        #init HED
        if m.weight.data.shape == torch.Size([1, 5, 1, 1]):
            # Constant initialization for fusion layer in HED network.
            torch.nn.init.constant_(m.weight, 0.2)
        else:
            # Zero initialization following official repository.
            # Reference: hed/docs/tutorial/layers.md
            m.weight.data.zero_()
        # Initialize: m.bias.
        if m.bias is not None:
            # Zero initialization.
            m.bias.data.zero_()


def resume(model, resume_path):
    if isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))


##========================== adjusting lrs

def tune_lrs(model, lr, weight_decay):

    bias_params= [param for name,param in list(model.named_parameters()) if name.find('bias')!=-1]
    weight_params= [param for name,param in list(model.named_parameters()) if name.find('weight')!=-1]


    if len(weight_params)==19:
        down1_4_weights , down1_4_bias  = weight_params[0:10]  , bias_params[0:10]
        down5_weights   , down5_bias    = weight_params[10:13] , bias_params[10:13]
        up1_5_weights    , up1_5_bias     = weight_params[13:18] , bias_params[13:18]
        fuse_weights , fuse_bias =weight_params[-1] , bias_params[-1]
        
        tuned_lrs=[
        {'params': down1_4_weights, 'lr': lr*1    , 'weight_decay': weight_decay},
        {'params': down1_4_bias,    'lr': lr*2    , 'weight_decay': 0.},
        {'params': down5_weights,   'lr': lr*100  , 'weight_decay': weight_decay},
        {'params': down5_bias,      'lr': lr*200  , 'weight_upecay': 0.},
        {'params': up1_5_weights,    'lr': lr*0.01 , 'weight_decay': weight_decay},
        {'params': up1_5_bias,       'lr': lr*0.02 , 'weight_decay': 0.},
        {'params': fuse_weights,    'lr': lr*0.001, 'weight_decay': weight_decay},
        {'params': fuse_bias ,      'lr': lr*0.002, 'weight_decay': 0.},
        ]

    elif len(weight_params)==32: #bn
        down1_4_weights , down1_4_bias  = weight_params[0:20]  , bias_params[0:20]
        down5_weights   , down5_bias    = weight_params[20:26] , bias_params[20:26]
        up1_5_weights    , up1_5_bias     = weight_params[26:31] , bias_params[26:31]
        fuse_weights , fuse_bias =weight_params[-1] , bias_params[-1]
        
        tuned_lrs=[
        {'params': down1_4_weights, 'lr': lr*1    , 'weight_decay': weight_decay},
        {'params': down1_4_bias,    'lr': lr*2    , 'weight_decay': 0.},
        {'params': down5_weights,   'lr': lr*100  , 'weight_decay': weight_decay},
        {'params': down5_bias,      'lr': lr*200  , 'weight_upecay': 0.},
        {'params': up1_5_weights,    'lr': lr*0.01 , 'weight_decay': weight_decay},
        {'params': up1_5_bias,       'lr': lr*0.02 , 'weight_decay': 0.},
        {'params': fuse_weights,    'lr': lr*0.001, 'weight_decay': weight_decay},
        {'params': fuse_bias ,      'lr': lr*0.002, 'weight_decay': 0.},
        ]
    else:
        print('Warning in tune_lrs')
        return model.parameters()

    
    return  tuned_lrs


##=========================== train_split func

def dev_checkpoint(save_dir, i, epoch, image_name, outputs):
    # display and logging
    if not isdir(save_dir):
        os.makedirs(save_dir)
    outs=[]
    for o in outputs:
        outs.append(tensor2image(o))
    if len(outs[-1].shape)==3:
        outs[-1]=outs[-1][0,:,:] #if RGB, show one layer only
    if i==-1:
        output_name=f"{image_name}.jpg"
    else:
        output_name=f"global_step-{i}-{image_name}.jpg"
    out=cv2.hconcat(outs) # if gray
    cv2.imwrite(join(save_dir, output_name), out)

def tensor2image(image):
            result = torch.squeeze(image.detach()).cpu().numpy()
            result = (result * 255).astype(np.uint8, copy=False)
            #(torch.squeeze(o.detach()).cpu().numpy()*255).astype(np.uint8, copy=False)
            return result

