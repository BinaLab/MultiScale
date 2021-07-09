import os, sys
import torch
#import numpy as np
#import scipy.io as sio
import argparse
from os.path import join, isfile #split, abspath, splitext, split, isdir, isfile



class struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



