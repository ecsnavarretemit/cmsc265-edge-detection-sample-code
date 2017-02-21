#!/usr/bin/env python

# sobel.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0

import os
import cv2
import numpy as np

image = os.path.join(os.getcwd(), "assets/images-small/indoor/DSC_0416.JPG")
# image = os.path.join(os.getcwd(), "assets/test/bikesgray.jpg")
cv_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

# depth should be set to -1 to automatically match the depth of the image source
# ksize = kernel size
sobel_x = cv2.Sobel(cv_image.copy(), -1, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
sobel_y = cv2.Sobel(cv_image.copy(), -1, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

# copy the contents of the matrix before modifying it
sobel_x_text = sobel_x.copy()
sobel_y_text = sobel_y.copy()

# add labels to the images
cv2.putText(sobel_x_text, "Sobel - X", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 190, 190), 3)
cv2.putText(sobel_y_text, "Sobel - Y", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 190, 190), 3)

# show the x and y image side by side
cv2.imshow('Sobel Operator Result', np.hstack([sobel_x_text, sobel_y_text]))

# converting back to uint8
abs_grad_x = cv2.convertScaleAbs(sobel_x.copy())
abs_grad_y = cv2.convertScaleAbs(sobel_y.copy())

# show the combined result
dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
cv2.imshow('Sobel Operator Combined Result', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()


