#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script provides functions for visualizing 1D & 2D tensors (array and matrix).
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def mat2img(matrix):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    im  = ax.imshow(matrix, cmap=cm.Greys_r) # , interpolation='nearest'
    fig.colorbar(im)
    plt.show()
