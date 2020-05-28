import argparse
from os.path import join, isfile #split, abspath, splitext, split, isdir, isfile


def arg_parser(dataset_path='../2012-main',pretrained_path='../pretrained_models/vgg',
                tmp='../tmp',resume_path=None,
                 data_reader=None, model=None,
                 batch_size=1, lr=1e-6, momentum=0.9, weight_decay=2e-4,stepsize=3, gamma=0.1,
                 start_epoch=0, maxepoch=10, itersize=10, print_freq=50,
                 gpu='0', use_cuda=False):

    parser = argparse.ArgumentParser(description='PyTorch Training')

    # ================ dataset and folders
#    parser.add_argument('--root', help='root folder', default=root)
    parser.add_argument('--dataset_path', help='root folder of dataset', default=dataset_path)
    parser.add_argument('--tmp', help='tmp folder', default=tmp)

    # ================= gpu and cuda
    parser.add_argument('--gpu', default=gpu, type=str, help='GPU ID')
    parser.add_argument('--use_cuda', default=use_cuda, type=str, help='Using Cuda')

    # ================ states
    parser.add_argument('--pretrained_path', default=pretrained_path, metavar='PATH',
                            help='path to the pretrained model (default: none)')
    parser.add_argument('--resume_path', default=resume_path, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    #================== training
    parser.add_argument('--data_reader', help='data reader class', default=data_reader)
    parser.add_argument('--model_type', help='model type', default=model)

    # =============== trainer
    parser.add_argument('--start_epoch', default=start_epoch, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--maxepoch', default=maxepoch, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--itersize', default=itersize, type=int,
                        metavar='IS', help='iter size')
    parser.add_argument('--stepsize', default=stepsize, type=int,
                        metavar='SS', help='learning rate step size')
    parser.add_argument('--batch_size', default=batch_size, type=int, metavar='BT',
                        help='batch size')
    # =============== optimizer
    parser.add_argument('--lr', '--learning_rate', default=lr, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=momentum, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=weight_decay, type=float,
                        metavar='W', help='default weight decay')

    parser.add_argument('--gamma', '--gm', default=gamma, type=float,
                        help='learning rate decay parameter: Gamma')

    # =============== misc

    parser.add_argument('--print_freq', '-p', default=print_freq, type=int,
                        metavar='N', help='print frequency (default: 50)')

    args=parser.parse_args()

    return args
