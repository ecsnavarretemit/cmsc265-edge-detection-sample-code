#!/usr/bin/env python

# laplacian.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0

import os
import cv2

image = os.path.join(os.getcwd(), "assets/images-small/indoor/DSC_0416.JPG")
# image = os.path.join(os.getcwd(), "assets/test/bikesgray.jpg")
cv_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

# ksize = kernel size
laplacian = cv2.Laplacian(cv_image, cv2.CV_64F, ksize=5)

cv2.imshow('Laplacian Operator Result', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()


