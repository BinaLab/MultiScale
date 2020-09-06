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

class Normalize_cv2(object):
    """normalize"""
    def __call__(self, img):
        img=np.array(img, dtype=np.float32)
        img -= np.array((104.00698793,116.66876762,122.67891434)) #rgb
        #img = np.transpose(im, (2, 0, 1)) # (H x W x C) to (C x H x W)
        return img

class Normalize_roi(object):

    def __call__(self,image):
        h1,w1=image.shape
        x, y, w, h = 0, h1//2, w1, h1//8
        ROI = image[y:y+h, x:x+w]

        # Calculate mean and STD
        mean, STD  = cv2.meanStdDev(ROI)

        # Clip frame to lower and upper STD
        offset = 10
        clipped = np.clip(image, mean - offset*STD, mean + offset*STD).astype(np.uint8)

        # Normalize to range
        #cv2.normalize(clipped, None?, 0, 255, norm_type=cv2.NORM_MINMAX)
        return cv2.normalize(clipped, clipped, 0, 255, norm_type=cv2.NORM_MINMAX)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        label = label - [left, top]

        return image , label


class Fliplr(object):
    """flip"""
    def __call__(self, image):
             
        return np.flip(image, axis=1)
    
class Flipud(object):
    """flip"""
    def __call__(self, image):
             
        return np.flip(image, axis=0)
    
# transform = transforms.Compose([
#      transforms.ToPILImage(),
#      transforms.Resize((300, 300)),
#      transforms.CenterCrop((100, 100)),
#      transforms.RandomCrop((80, 80)),
#      transforms.RandomHorizontalFlip(p=0.5),
#      transforms.RandomRotation(degrees=(-90, 90)),
#      transforms.RandomVerticalFlip(p=0.5),
#      transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#      ])
    
