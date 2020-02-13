#!/usr/bin/env python

"""
The stacks module contains common functions used when
manipulating stacks with python
"""

__all__ = ['releaf', 'read_stack_to_channels','interleaf']

__version__ = "0.1.0"

__author__ = 'Sy Redding'

import os
import numpy as np
import warnings
from smtools.alignment import im_split
from skimage.external.tifffile import imread


def releaf(Image1, Image2):
    """

    :param Image1:
    :param Image2:
    :return:
    """
    _,ch1_right = im_split(Image1)
    ch2_left,_ = im_split(Image2)
    return(np.concatenate([ch2_left,ch1_right], axis = 1))

def outerleaf(Image_list):
    if len(Image_list)%2 == 1:
        warnings.simplefilter('always')
        warnings.warn("Warning passed to `outerleaf` contains odd number of images, middle image dropped")
        warnings.simplefilter('ignore')
        return Image_list[:(len(Image_list)//2)-1],Image_list[len(Image_list)//2:]
    else:
        return Image_list[:len(Image_list)//2],Image_list[len(Image_list)//2:]

def interleaf(stack_0, stack_1, invert=False):
    """

    :param stack_0:
    :param stack_1:
    :param invert:
    :return:
    """
    ch0_list, ch1_list = [], []
    for i, j in zip(stack_0, stack_1):
        if invert:
            ch0, _ = im_split(i)
            _, ch1 = im_split(j)
        else:
            _, ch0 = im_split(i)
            ch1, _ = im_split(j)

        ch0_list.append(ch0)
        ch0_list.append(ch0)
        ch1_list.append(ch1)
        ch1_list.append(ch1)

    ch0_list.pop(0)
    ch1_list.pop(-1)

    stretched_and_combined = []
    for i, j in zip(ch0_list, ch1_list):
        stretched_and_combined.append(np.concatenate([i, j], axis=1))
    return (stretched_and_combined)



def read_stack_to_channels(directory):
    """
    requires digital modulation
    :param directory:
    :return:
    """
    ch0_names, ch1_names = [], []
    for i in os.listdir(directory):
        if i.endswith(".tif"):
            if i.split("_")[1][-1] == "0":
                ch0_names.append(i)
            if i.split("_")[1][-1] == "1":
                ch1_names.append(i)
    ch0_names.sort()
    ch1_names.sort()

    ch0_stack, ch1_stack = [], []
    for i, j in zip(ch0_names, ch1_names):
        ch0_stack.append(imread(directory + i))
        ch1_stack.append(imread(directory + j))

    return (ch0_stack, ch1_stack)
