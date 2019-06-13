import sys # system functions (ie. exiting the program)
import os # operating system functions (ie. path building on Windows vs. MacOs)
import time # for time operations
import uuid # for generating unique file names
import math # math functions

from IPython.display import display as ipydisplay, Image, clear_output, HTML # for interacting with the notebook better

import numpy as np # matrix operations (ie. difference between two matricies)
import cv2 # (OpenCV) computer vision functions (ie. tracking)
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))

import matplotlib.pyplot as plt # (optional) for plotting and showing images inline

#import keras # high level api to tensorflow (or theano, CNTK, etc.) and useful image preprocessing
#from keras import backend as K
#from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
#from keras.models import Sequential, load_model, model_from_json
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#print('Keras image data format: {}'.format(K.image_data_format()))

from picamera.array import PiRGBArray
from picamera import PiCamera

def show_image(image, inv=False):
	if inv:
		corrected_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	else:
		corrected_image = image
	cv2.imshow("Image", corrected_image)
	cv2.waitKey(0)
	#Destroy Window
	cv2.destroyAllWindows()

camera = PiCamera()
rawCapture = PiRGBArray(camera)

time.sleep(0.5)

camera.resolution = (640, 480)
camera.capture(rawCapture, format="bgr")
camera.close()

bgr_image = rawCapture.array
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

show_image(bgr_image)
# show_image(rgb_image, inv=True)

blur_img = bgr_image.copy()
blur_img = cv2.GaussianBlur(blur_img, (41, 41), 10)
show_image(blur_img)

dilate_img = bgr_image.copy()
dilate_img = cv2.dilate(dilate_img, np.ones((10,10), dtype=np.uint8), iterations=1)
show_image(dilate_img)

erosion_img = bgr_image.copy()
erosion_img = cv2.erode(erosion_img, np.ones((10,10), dtype=np.uint8), iterations=1)
show_image(erosion_img)

canny_img = bgr_image.copy()
canny_img = cv2.erode(canny_img, np.ones((8,8), dtype=np.uint8), iterations=1)
edges = cv2.Canny(canny_img,100,100)
show_image(edges.astype(np.uint8))

thresh_img = bgr_image.copy()
thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(thresh_img, 80, 255, cv2.THRESH_BINARY)
show_image(thresh)
