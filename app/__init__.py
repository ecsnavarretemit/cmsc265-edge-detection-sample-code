# __init__.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import numpy as np
from scipy.ndimage import filters

def apply_prewitt(img):
  img = img.copy()

  imx = np.zeros(img.shape)
  filters.prewitt(img, 1, imx, mode="nearest")

  imy = np.zeros(img.shape)
  filters.prewitt(img, 0, imy, mode="nearest")

  #the magnitude
  grad = np.sqrt(imx ** 2 + imy ** 2)

  return imx, imy, grad

def apply_prewitt2(img):
  img = img.copy()

  dx = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0],])
  dy = np.transpose(dx)

  fo1 = filters.convolve(img, dx, output=np.float64, mode='nearest')
  fo2 = filters.convolve(img, dy, output=np.float64, mode='nearest')

  return fo1, fo2

def apply_roberts(img):
  img = img.copy()

  # assemble the roberts kernel
  dx = np.array([[1, 0], [0, -1]])
  dy = np.array([[0, -1], [1, 0]])

  # apply the x and y kernels
  fo1 = filters.convolve(img, dx, output=np.float64, mode='nearest')
  fo2 = filters.convolve(img, dy, output=np.float64, mode='nearest')

  return fo1, fo2


