#!/usr/bin/env python

# prewitt.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import numpy as np
from app import apply_prewitt2

image = os.path.join(os.getcwd(), "assets/images-small/indoor/DSC_0416.JPG")
# image = os.path.join(os.getcwd(), "assets/test/bikesgray.jpg")
cv_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

prewitt_x, prewitt_y = apply_prewitt2(cv_image)

# copy the contents of the matrix before modifying it
prewitt_x_text = prewitt_x.copy()
prewitt_y_text = prewitt_y.copy()

# add labels to the images
cv2.putText(prewitt_x_text, "Prewitt - X", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 190, 190), 3)
cv2.putText(prewitt_y_text, "Prewitt - Y", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 190, 190), 3)

# show the x and y image side by side
cv2.imshow('Prewitt Operator Result', np.hstack([prewitt_x_text, prewitt_y_text]))

# converting back to uint8
abs_grad_x = cv2.convertScaleAbs(prewitt_x.copy())
abs_grad_y = cv2.convertScaleAbs(prewitt_y.copy())

# show the combined result
dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
cv2.imshow('Prewitt Operator Combined Result', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()


