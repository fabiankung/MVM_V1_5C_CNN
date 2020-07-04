# -*- coding: utf-8 -*-
"""
Last Modified:      3 June 2020
Author:             Fabian Kung


A simple CNN model that analyze a 120x160 pixels grayscale image to 
determine if there is any object on the floor.  The main assumption:
    The camera is always pointed towards the floor, so we only need
    to analyze the lower half of the image.
    
Due to speed and memory limitation of the CPU (the inference model 
is to be run on a 32-bits micro-controller), we further reduce the 
size of the image to be analyzed.  Here only a region-of-interest (ROI)
size of 37x100 pixels in the lower half of the image is analyzed.

The file:
    1. Loads a bitmap (BMP) images for training and test samples and generates
    a 2D numpy array for each image, and then crop out the ROI of the image.
    2. Trains and verify the CNN using the images.  The CNN has four outputs, 
       Obstacle on left
       Obstacle on right
       Obstacle in front
       No Obstacle/object
    3. Perform inference operation on a selected image.
    4. Export the weights and topology of the CNN into either a text file or 
    a C/C++ header file, for incorporation into an edge processor.
    
@author: User
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

tf.keras.backend.clear_session()  # For easy reset of notebook state.


#Set the width and height of the input image in pixels.
_imgwidth = 160
_imgheight = 120

#Set the region of interest start point and size.
#Note: The coordinate (0,0) starts at top left hand corner of the image frame.
_roi_startx = 30
_roi_starty = 71
_roi_width = 100
_roi_height = 37

#CNN model parameters
_layer0_channel = 16   #No. of channels in 1st layer, e.g. no. of 2D convolution filter.
_DNN1_node = 35 #No. of nodes in 2nd layer.
_DNN2_node = 4  #No. of nodes in 3rd layer (Output).


train_dir = os.path.join('./TrainImage/NoObject') #Create a path to folder the no object in training image directory
train_names = os.listdir(train_dir)   #Create a list containing the filenames of all image files in the directory.
print("Training file names, 'no object': ", train_names)
print("")
train_num_files = len(os.listdir(train_dir))

train_dir2 = os.path.join('./TrainImage/Left') #Create a path to the folder with object on left in training image directory
train_names2 = os.listdir(train_dir2)   #Create a list containing the filenames of all image files in the directory.
print("Training file names, 'With object on left': ", train_names2)
print("")
train_num_files2 = len(os.listdir(train_dir2))

train_dir3 = os.path.join('./TrainImage/Right') #Create a path to the folder with object on right in training image directory
train_names3 = os.listdir(train_dir3)   #Create a list containing the filenames of all image files in the directory.
print("Training file names, 'With object on right': ", train_names3)
print("")
train_num_files3 = len(os.listdir(train_dir3))

train_dir4 = os.path.join('./TrainImage/Front') #Create a path to the folder with object in front in training image directory
train_names4 = os.listdir(train_dir4)   #Create a list containing the filenames of all image files in the directory.
print("Training file names, 'With object in front': ", train_names4)
print("")
train_num_files4 = len(os.listdir(train_dir4))

#--- Load training images and attach label ---

#Create an empty 3D array to hold the sequence of 2D image data and
#1D array to hold the labels

train_images = np.empty([train_num_files + train_num_files2 + train_num_files3
                         + train_num_files4,_roi_height,_roi_width])
train_labels = np.empty([train_num_files + train_num_files2 + train_num_files3
                         + train_num_files4])


#Read BMP file, extract grayscale value, crop and fill into train_images
#Note: This can also be done using keras.image class, specifically the
#keras.image.load_image() and keras.image.img_to_array() methods.
i = 0
for train_image_file in train_names:  #Training images, no object.
    #Read original BMP image
    image = plt.imread(train_dir+'/'+train_image_file,format = 'BMP') #Can also use os.join function to create the complete filename.
    #Extract only 1 channel of the RGB data, assign to 2D array
    imgori = image[0:_imgheight,0:_imgwidth,0]
    #Crop the 2D array to only the region of interest
    train_images[i] = imgori[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width]
    #Fill up the label array (here each class has 5 samples)
    train_labels[i] = 0  #Label value for no object.
    i = i+1
    #Plot the training images
    #plt.figure(num=i) #1 image frame per figure.
    #plt.imshow(train_images[i-1],cmap='gray')
    
for train_image_file in train_names2:  #Training images, with object on left.
    #Read original BMP image
    image = plt.imread(train_dir2+'/'+train_image_file,format = 'BMP') #Can also use os.join function to create the complete filename.
    #Extract only 1 channel of the RGB data, assign to 2D array
    imgori = image[0:_imgheight,0:_imgwidth,0]
    #Crop the 2D array to only the region of interest
    train_images[i] = imgori[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width]
    #Fill up the label array (here each class has 5 samples)
    train_labels[i] = 1  #Label value for object on left.
    i = i+1
    #Plot the training images
    #plt.figure(num=i) #1 image frame per figure.
    #plt.imshow(train_images[i-1],cmap='gray')
    

for train_image_file in train_names3:  #Training images, with object on right.
    #Read original BMP image
    image = plt.imread(train_dir3+'/'+train_image_file,format = 'BMP') #Can also use os.join function to create the complete filename.
    #Extract only 1 channel of the RGB data, assign to 2D array
    imgori = image[0:_imgheight,0:_imgwidth,0]
    #Crop the 2D array to only the region of interest
    train_images[i] = imgori[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width]
    #Fill up the label array (here each class has 5 samples)
    train_labels[i] = 2  #Label value for object on right.
    i = i+1
    #Plot the training images
    #plt.figure(num=i) #1 image frame per figure.
    #plt.imshow(train_images[i-1],cmap='gray')
     
for train_image_file in train_names4:  #Training images, with object in front.
    #Read original BMP image
    image = plt.imread(train_dir4+'/'+train_image_file,format = 'BMP') #Can also use os.join function to create the complete filename.
    #Extract only 1 channel of the RGB data, assign to 2D array
    imgori = image[0:_imgheight,0:_imgwidth,0]
    #Crop the 2D array to only the region of interest
    train_images[i] = imgori[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width]
    #Fill up the label array (here each class has 5 samples)
    train_labels[i] = 3  #Label value for object in front.
    i = i+1
    #Plot the training images
    #plt.figure(num=i) #1 image frame per figure.
    #plt.imshow(train_images[i-1],cmap='gray')
    
    
#--- Load test images and attach label ---
test_dir = os.path.join('./TestImage/NoObject')
test_names = os.listdir(test_dir)
print("Test file names, 'no object': ", test_names)
print("")
test_num_files = len(os.listdir(test_dir))

test_dir2 = os.path.join('./TestImage/Left')
test_names2 = os.listdir(test_dir2)
print("Test file names, 'with object on left': ", test_names2)
print("")
test_num_files2 = len(os.listdir(test_dir2))

test_dir3 = os.path.join('./TestImage/Right')
test_names3 = os.listdir(test_dir3)
print("Test file names, 'with object on right': ", test_names3)
print("")
test_num_files3 = len(os.listdir(test_dir3))

test_dir4 = os.path.join('./TestImage/Front')
test_names4 = os.listdir(test_dir4)
print("Test file names, 'with object in front': ", test_names4)
print("")
test_num_files4 = len(os.listdir(test_dir4))


#Read BMP file, extract grayscale value, crop and fill into train_images

#Create an empty 3D array to hold the sequence of 2D image data and labels

test_images = np.empty([test_num_files + test_num_files2 + test_num_files3 + 
                        test_num_files4,_roi_height,_roi_width])
test_labels = np.empty([test_num_files + test_num_files2 + test_num_files3 +
                        test_num_files4])

i = 0
for test_image_file in test_names:  #Test images, no object.
    #Read original BMP image
    image = plt.imread(test_dir+'/'+test_image_file,format = 'BMP')
    #Extract only 1 channel of the RGB data, assign to 2D array
    imgori = image[0:_imgheight,0:_imgwidth,0]
    #Crop the 2D array to only the region of interest
    test_images[i] = imgori[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width]
    #Fill up the label array (here each class has 20 samples)
    test_labels[i] = 0  #Label value for no object.
    i = i+1    
    #Plot the test images
    #plt.figure(num=i) #1 frame per figure.
    #plt.imshow(test_images[i-1],cmap='gray')
    
for test_image_file in test_names2:  #Test images, with object on left.
    #Read original BMP image
    image = plt.imread(test_dir2+'/'+test_image_file,format = 'BMP')
    #Extract only 1 channel of the RGB data, assign to 2D array
    imgori = image[0:_imgheight,0:_imgwidth,0]
    #Crop the 2D array to only the region of interest
    test_images[i] = imgori[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width]
    #Fill up the label array (here each class has 20 samples)
    test_labels[i] = 1  #Label value for with object on left.
    i = i+1    
    #Plot the test images
    #plt.figure(num=i) #1 frame per figure.
    #plt.imshow(test_images[i-1],cmap='gray')

for test_image_file in test_names3:  #Test images, with object.
    #Read original BMP image
    image = plt.imread(test_dir3+'/'+test_image_file,format = 'BMP')
    #Extract only 1 channel of the RGB data, assign to 2D array
    imgori = image[0:_imgheight,0:_imgwidth,0]
    #Crop the 2D array to only the region of interest
    test_images[i] = imgori[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width]
    #Fill up the label array (here each class has 20 samples)
    test_labels[i] = 2  #Label value for with object on right.
    i = i+1    
    #Plot the test images
    #plt.figure(num=i) #1 frame per figure.
    #plt.imshow(test_images[i-1],cmap='gray')

for test_image_file in test_names4:  #Test images, with object in front.
    #Read original BMP image
    image = plt.imread(test_dir4+'/'+test_image_file,format = 'BMP')
    #Extract only 1 channel of the RGB data, assign to 2D array
    imgori = image[0:_imgheight,0:_imgwidth,0]
    #Crop the 2D array to only the region of interest
    test_images[i] = imgori[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width]
    #Fill up the label array (here each class has 20 samples)
    test_labels[i] = 3  #Label value for with object in front.
    i = i+1    
    #Plot the test images
    #plt.figure(num=i) #1 frame per figure.
    #plt.imshow(test_images[i-1],cmap='gray')

train_images = train_images/256.0 #Normalize the training image array, and 
                                  #convert to floating point.
                                  #Alternatively we can use:
#train_images = train_images.astype('float32')/256                                      
train_images=train_images.reshape(train_num_files + train_num_files2 + train_num_files3 +
                                  train_num_files4, _roi_height, _roi_width, 1)

test_images = test_images/256.0 #Normalize the test image array.
test_images=test_images.reshape(test_num_files + test_num_files2 + test_num_files3 +
                                test_num_files4, _roi_height, _roi_width, 1)


# Model - CNN with single convolution layer and single max-pooling layer, 2 DNN layers.
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(_layer0_channel, (3,3), strides = 2, activation='relu', input_shape=(_roi_height, _roi_width, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(_DNN1_node, activation='relu'),
    tf.keras.layers.Dense(_DNN2_node, activation='softmax')
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Optional, generating a plot of the model. Requires pydot and graphviz to be
# installed. If the path to the graphic file "my_first_model.png" is not given, it 
# will be saved to the same folder as this sourcecode.
tf.keras.utils.plot_model(model, 'CNN_model.png', show_shapes = True) 

history = model.fit(train_images, train_labels, epochs=30)

model.evaluate(test_images, test_labels)


"""
#--- Create a CNN for prediction, and analyze 1 test image ---
"""

from tensorflow.keras import models
import matplotlib.pyplot as plt
f, axs = plt.subplots(4,4)
f2, axs2 = plt.subplots(4,4)


# Here we use the list comprehension feature of Python, to create a list from an
# an old list which contains all the layers in the CNN model.  Note that list 
# comprehension syntax actually follows typical mathematical notation to form a list.
layer_outputs = [layer.output for layer in model.layers] # Get the output from each layer
                                                         # in the CNN and form a list
                                                         # or a sequence of layer output.
                                                         
# Create a new tensorflow model for inference or prediction.  Note that here we are 
# using the keras functional approach to construct the CNN model, as opposed to the 
# sequential approach when creating the training model.                                                     
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)                                                         

#image = plt.imread('ImgColorBar.bmp',format = 'BMP')  # Load BMP image to be analyzed by the CNN.
image = plt.imread('2.bmp',format = 'BMP')
image = image/256.0  # Normalized to between 0 to 1.0.
#Extract only 1 channel of the RGB data, assign to 2D array
imgori = image[0:_imgheight,0:_imgwidth,0]
#Crop the 2D array to only the region of interest
imgcrop = imgori[_roi_starty:_roi_starty+_roi_height,_roi_startx:_roi_startx+_roi_width]


resultlayer0 = activation_model.predict(imgcrop.reshape(1,_roi_height,_roi_width,1))[0] # Result after Conv2D.
resL0fil0 = resultlayer0[0,:,:,0] 
resL0fil1 = resultlayer0[0,:,:,1] 
resL0fil2 = resultlayer0[0,:,:,2] 
resL0fil3 = resultlayer0[0,:,:,3] 
resultlayer1 = activation_model.predict(imgcrop.reshape(1,_roi_height,_roi_width,1))[1] # Result after Max Pooling.
resultlayer2 = activation_model.predict(imgcrop.reshape(1,_roi_height,_roi_width,1))[2] # Result after flatten.
resultlayer3 = activation_model.predict(imgcrop.reshape(1,_roi_height,_roi_width,1))[3] # Result after DNN.
resultlayer4 = activation_model.predict(imgcrop.reshape(1,_roi_height,_roi_width,1))[4] # Result of output layer.
print("Classify 1 test sample")
print(resultlayer4)
#Plot the output of each convolution filter and maxpool output in a 4x4 grid.
for x in range(0,16):
    div = int(x/4)
    mod = int(x%4)
    axs[div,mod].imshow(resultlayer0[0,:,:,x],cmap='plasma')
    axs[div,mod].grid = (False)
    
for x in range(0,16):
    div = int(x/4)
    mod = int(x%4)
    axs2[div,mod].imshow(resultlayer1[0,:,:,x],cmap='plasma')
    axs2[div,mod].grid = (False)    



"""
#--- Save the weights into a text file ---
"""
# Get the weights
wt = model.get_weights()
# The highest level index to wt points to the coefficients or weights of each layer.
wtConv2D1 = wt[0]        # Weights of 1st convolution layer.
wtConv2D1bias = wt[1]    # Bias of 1st convolution layer.
wtDNN1 = wt[2]          # Weights of 1st DNN layer.
wtDNN1bias = wt[3]      # Bias of 1st DNN layer.
wtDNN2 = wt[4]          # Weights of 2nd DNN layer.
wtDNN2bias = wt[5]          # Bias of 2nd DNN layer.

Conv2D1filter = wtConv2D1.shape[3]  # get no. of filters in 1st convolution layer.
Flattennode = wtDNN1.shape[0]       # get no. of nodes after flatting the convolutional layer.
DNN1node = wtDNN1.shape[1]          # get no. of nodes in 1st DNN layer.
DNN2node = wtDNN2.shape[1]          # get no. of nodes in 2nd DNN layer.

# Open the file.
f = open("C:\CNN.h","w+")     # Header file to store the coefficients.

# Set the parameters of the filter and other constants in the CNN.


f.write("#define  __ROI_STARTX  %d \n" % _roi_startx)
f.write("#define  __ROI_STARTY  %d \n" % _roi_starty)
f.write("#define  __ROI_WIDTH  %d \n" % _roi_width)
f.write("#define  __ROI_HEIGHT  %d \n" % _roi_height)
f.write("#define  __FILTER_SIZE  3 \n")
f.write("#define  __FILTER_STRIDE  2 \n")
f.write("#define  __LAYER0_CHANNEL  %d \n" % _layer0_channel)
f.write("#define  __LAYER0_X  %d \n" % ((_roi_width-3)/2 + 1))
f.write("#define  __LAYER0_Y  %d \n" % ((_roi_height-3)/2 + 1))
f.write("#define  __DNN1NODE %d \n" % _DNN1_node)
f.write("#define  __DNN2NODE %d" % _DNN2_node)
f.write("\n\n")
# Conv2D1
N = 3   # Filter size, 3x3

f.write("const  int  gnL1f[%d][%d][%d] = { \n" % (Conv2D1filter,N,N))  # Integer version
for nfilter in range(Conv2D1filter):
    f.write("{")
    for i in range(3):
        f.write("{")
        for j in range(3):
            f.write("%d" % (wtConv2D1[i,j,0,nfilter]*1000000))  # Scaled integer version.
            if j < (N-1):
                f.write(", ")       # Add a comma and space after every number, except last number.
        f.write("}")        
        if i < (N-1):
            f.write(", ")
    if nfilter < (Conv2D1filter - 1):
        f.write("}, \n")
    else:
        f.write("} \n")
f.write("}; \n\n")

# Bias for Conv2D1
f.write("const  int  gnL1fbias[%d] = {" % Conv2D1filter)  # Integer version
for nfilter in range(Conv2D1filter):
    f.write("%d" % (wtConv2D1bias[nfilter]*1000000))  # Scaled integer version
    if nfilter < (Conv2D1filter-1):
        f.write(", ")
f.write("}; \n\n")

# DNN layer 1
# Weights
f.write("const  int  gnDNN1w[%d][%d] = { \n" % (Flattennode,DNN1node))      # Integer version.
for i in range(Flattennode):
    f.write("{")
    for j in range(DNN1node):
        f.write("%d" % (wtDNN1[i,j]*1000000)) # Scaled integer version.
        if j < (DNN1node - 1):
            f.write(", ")           # Add a comma and space after every number, except last number.
    if i < (Flattennode - 1):
        f.write("},\n")             # Add a newline and '}' after every row.
    else:
        f.write("} \n")
f.write("}; \n\n")

# Bias
f.write("const int gnDNN1bias[%d] = {" % DNN1node)  # Scaled integer veresion.
for i in range(DNN1node):
    f.write("%d" % (wtDNN1bias[i]*1000000))  #Scaled to integer.
    if i < (DNN1node - 1):
        f.write(", ")
f.write("}; \n\n")

# DNN layer 2
# Weights
f.write("const  int  gnDNN2w[%d][%d] = { \n" % (DNN1node,DNN2node))  # Scaled integer veresion.
for i in range(DNN1node):
    f.write("{")
    for j in range(DNN2node):
        f.write("%d" % (wtDNN2[i,j]*1000000))  # Scaled integer veresion.
        if j < (DNN2node - 1):
            f.write(", ")           # Add a comma and space after every number, except last number.
    if i < (DNN1node - 1):
        f.write("},\n")             # Add a newline and '}' after every row. 
    else:
        f.write("} \n")
f.write("}; \n\n")

# Bias
f.write("const int gnDNN2bias[%d] = {" % DNN2node)
for i in range(DNN2node):
    f.write("%d" % (wtDNN2bias[i]*1000000))  # Scaled to integer veresion.
    if i < (DNN2node - 1):
        f.write(", ")
f.write("}; \n\n")

f.close()
