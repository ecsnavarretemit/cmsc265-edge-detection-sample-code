#!/usr/bin/env python

# roberts.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import numpy as np
from app import apply_roberts

image = os.path.join(os.getcwd(), "assets/images-small/indoor/DSC_0416.JPG")
# image = os.path.join(os.getcwd(), "assets/test/bikesgray.jpg")
cv_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

roberts_x, roberts_y = apply_roberts(cv_image)

# copy the contents of the matrix before modifying it
roberts_x_text = roberts_x.copy()
roberts_y_text = roberts_y.copy()

# add labels to the images
cv2.putText(roberts_x_text, "Roberts - X", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 190, 190), 3)
cv2.putText(roberts_x_text, "Roberts - Y", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 190, 190), 3)

# show the x and y image side by side
cv2.imshow('Roberts Operator Result', np.hstack([roberts_x_text, roberts_x_text]))

# converting back to uint8
abs_grad_x = cv2.convertScaleAbs(roberts_x.copy())
abs_grad_y = cv2.convertScaleAbs(roberts_y.copy())

# show the combined result
dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
cv2.imshow('Roberts Operator Combined Result', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()


