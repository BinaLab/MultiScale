# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:04:46 2020

@author: yari
"""
import cv2
import numpy as np

class Rescale_size(object):
    """Rescale."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
    def __call__(self, image):
        return cv2.resize(image, self.output_size)

class Rescale_byrate(object):
    """Rescale."""

    def __init__(self, output_size_rate):
        assert isinstance(output_size_rate, (int, tuple, float))
        if isinstance(output_size_rate, float):
            self.output_size_rate = (output_size_rate, output_size_rate)
        else:
            assert len(output_size_rate) == 2
            self.output_size_rate = output_size_rate
    def __call__(self, image):
        h , w =image.shape[0], image.shape[1]
        return cv2.resize(image, (int(w*self.output_size_rate[0]) , int(h*self.output_size_rate[1])))


class Fliplr(object):
    """flip"""
    def __call__(self, image):

        return np.flip(image, axis=1)

class Flipud(object):
    """flip"""
    def __call__(self, image):

        return np.flip(image, axis=0)
