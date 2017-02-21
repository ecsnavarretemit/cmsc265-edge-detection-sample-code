# __init__.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0

import numpy as np
from scipy.ndimage import filters

def apply_prewitt(img):
  img = img.copy()

  im_x = np.zeros(img.shape)
  filters.prewitt(img, 1, im_x, mode="nearest")

  im_y = np.zeros(img.shape)
  filters.prewitt(img, 0, im_y, mode="nearest")

  #the magnitude
  grad = np.sqrt(im_x ** 2 + im_y ** 2)

  return im_x, im_y, grad

def apply_prewitt2(img):
  img = img.copy()

  dx = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0],])
  dy = np.transpose(dx)

  im_x = filters.convolve(img, dx, output=np.float64, mode='nearest')
  im_y = filters.convolve(img, dy, output=np.float64, mode='nearest')

  return im_x, im_y

def apply_roberts(img):
  img = img.copy()

  # assemble the roberts kernel
  dx = np.array([[1, 0], [0, -1]])
  dy = np.array([[0, -1], [1, 0]])

  # apply the x and y kernels
  im_x = filters.convolve(img, dx, output=np.float64, mode='nearest')
  im_y = filters.convolve(img, dy, output=np.float64, mode='nearest')

  return im_x, im_y


