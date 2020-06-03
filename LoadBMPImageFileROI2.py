# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 11:42:21 2020
The file loads a bitmap image and generate a 2D numpy array for
the image, and then crop out the region-of-interest of the image
file.  Sometimes only a part of the image is used for training, so that
we can use a smaller NN.  The part of the image used for training is the
ROI (region-of-interest), and the boundary will be highlighted in the
image display.
@author: User
"""
import matplotlib.pyplot as plt

#Set the actual width and height of the image in pixels.
_imgwidth = 160
_imgheight = 120

#Set the region-of-interest (ROI) start point and size.
_roi_startx = 30
_roi_starty = 71
_roi_width = 100
_roi_height = 37

#Load the BMP image, the image is assumed to contain RGB channels, but here 
#it is a grayscale image so all R, G and B channels have similar values of
#between 0-255.
#The filename and path of the bitmap image file is fixed.
print("Loading image...")
image = plt.imread('12.bmp',format = 'BMP')


#Extract only 1 channel of the pixel, since all RGB channels have the same
#values.  Also add the boundary of the ROI (in black).
imgori = image[0:_imgheight,0:_imgwidth,0]


print("Shape of image is ", imgori.shape)
print("Original image")

plt.figure(num=1)  #Create figure frame for plotting the original image.
plt.imshow(imgori,cmap='gray') #Show grayscale image, cmap means colormap.

#Add boundaries for ROI in image.
plt.axvline(_roi_startx,1 - (_roi_starty + _roi_height)/_imgheight,
            1 - _roi_starty/_imgheight,color='red')
plt.axvline(_roi_startx + _roi_width/3,1 - (_roi_starty + _roi_height)/_imgheight,
            1 - _roi_starty/_imgheight,color='red')
plt.axvline(_roi_startx + (2*_roi_width)/3,1 - (_roi_starty + _roi_height)/_imgheight,
            1 - _roi_starty/_imgheight,color='red')
plt.axvline(_roi_startx + _roi_width,1 - (_roi_starty + _roi_height)/_imgheight,
            1 - _roi_starty/_imgheight,color='red')
plt.axhline(_roi_starty,_roi_startx/_imgwidth,
            (_roi_startx + _roi_width)/_imgwidth,color='red')
plt.axhline(_roi_starty + _roi_height,_roi_startx/_imgwidth,
            (_roi_startx + _roi_width)/_imgwidth,color='red')
plt.show()

#Crop the image as per the parameters in the ROI.
print("The region-of-interest (ROI)")
imgcrop = imgori[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width]
print("Shape of the ROI is ", imgcrop.shape)
plt.figure(num=2)  #Create figure 2 for plotting the cropped image.  Note: If we
                   #did not do this, then the last figure frame will be cleared,
                   #and a new figure will be plot in it.
plt.imshow(imgcrop,cmap='gray')