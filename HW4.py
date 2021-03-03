#!/usr/bin/env python
# coding: utf-8

# # HW: X-ray images classification
# --------------------------------------

# Before you begin, open Mobaxterm and connect to triton with the user and password you were give with. Activate the environment `2ndPaper` and then type the command `pip install scikit-image`.

# In this assignment you will be dealing with classification of 32X32 X-ray images of the chest. The image can be classified into one of four options: lungs (l), clavicles (c), and heart (h) and background (b). Even though those labels are dependent, we will treat this task as multiclass and not as multilabel. The dataset for this assignment is located on a shared folder on triton (`/MLdata/MLcourse/X_ray/'`).

# In[ ]:


import os
import numpy as np
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.layers import Flatten, InputLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import *

from tensorflow.keras.initializers import Constant
from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from skimage.io import imread

from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"


# In[ ]:


import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# In[ ]:


def preprocess(datapath):
    # This part reads the images
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        src = imread(os.path.join(datapath, fn),1)
        img = resize(src,(32,32),order = 3)
        
        images[ii,:,:,0] = img
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    BaseImages = images
    BaseY = Y
    return BaseImages, BaseY


# In[ ]:


def preprocess_train_and_val(datapath):
    # This part reads the images
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        images[ii,:,:,0] = imread(os.path.join(datapath, fn),1)
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    return images, Y


# In[ ]:


#Loading the data for training and validation:
src_data = '/MLdata/MLcourse/X_ray/'
train_path = src_data + 'train'
val_path = src_data + 'validation'
test_path = src_data + 'test'
BaseX_train , BaseY_train = preprocess_train_and_val(train_path)
BaseX_val , BaseY_val = preprocess_train_and_val(val_path)
X_test, Y_test = preprocess(test_path)


# In[ ]:


keras.backend.clear_session()


# ### PART 1: Fully connected layers 

# Elaborate a NN with 2 hidden fully connected layers with 300, 150 neurons and 4 neurons for classification. Use ReLU activation functions for the hidden layers and He_normal for initialization. Don't forget to flatten your image before feedforward to the first dense layer. Name the model `model_relu`.*

#--------------------------Implement your code here:-------------------------------------

X_train = BaseX_train
Y_train = BaseY_train
X_val = BaseX_val
Y_val = BaseY_val

# Setting up the NN (ReLU)
dropout=0.2 #for this Part, we regularize using dropout
model_relu = Sequential(name="model_relu")
model_relu.add(Flatten(input_shape=[32, 32, 1]))
model_relu.add(Dense(300, kernel_initializer="he_normal", activation='relu'))
model_relu.add(Dropout(dropout))
model_relu.add(Dense(150, kernel_initializer="he_normal", activation='relu'))
model_relu.add(Dropout(dropout))
model_relu.add(Dense(4, kernel_initializer="he_normal", activation='softmax'))
#----------------------------------------------------------------------------------------


model_relu.summary()


#Inputs: 
#input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


# Compile the model with the optimizer above, accuracy metric and adequate loss for multiclass task. Train your model on the training set and evaluate the model on the testing set. Print the accuracy and loss over the testing set.

#--------------------------Impelment your code here:-------------------------------------
# Compiling the NN model
model_relu.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)

# Saving initial weights
if not ("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "init_weights_ReLU64.h5"
model_path = os.path.join(save_dir, model_name)
model_relu.save(model_path)
print('Saved initialized ReLU model at %s ' % model_path)


# Fitting the model to the training data and evaluating with the testing set
history = model_relu.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,verbose=0,
                         validation_data=(X_test, Y_test))

# Plotting the loss & accuracy (with regards to the testing set)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(history.history['acc'])
axs[0].plot(history.history['val_acc'])
axs[0].set_title('ReLU-Model Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(['Train', 'Test'], loc='lower right')

axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('ReLU-Model Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(['Train', 'Test'], loc='upper right')

plt.tight_layout()
plt.show()

# Evaluating model
loss_and_metrics = model_relu.evaluate(X_test, Y_test, verbose=2)
#----------------------------------------------------------------------------------------

# Change the activation functions to LeakyRelu or tanh or sigmoid. Name the new model `new_a_model`. Explain how it can affect the model.*

#We chose TANH
#--------------------------Impelment your code here:-------------------------------------

new_a_model = Sequential(name="new_a_model")
new_a_model.add(Flatten(input_shape=[32, 32, 1]))
new_a_model.add(Dense(300, kernel_initializer="he_normal", activation='tanh'))
new_a_model.add(Dropout(dropout))
new_a_model.add(Dense(150, kernel_initializer="he_normal", activation='tanh'))
new_a_model.add(Dropout(dropout))
new_a_model.add(Dense(4, kernel_initializer="he_normal", activation='softmax'))
#----------------------------------------------------------------------------------------


new_a_model.summary()



# Train the new model using 25 and 40 epochs. What difference does it makes in term of performance? Remember to save the compiled model for having initialized weights for every run as we did in tutorial 12. Evaluate each trained model on the test set*

#-----------------------25 EPOCHS-----------------------------------------------

#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25


#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


#--------------------------Impelment your code here:-------------------------------------
# Compiling the model
new_a_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)

# Saving the initial weights
save_dir = "results/"
model_name = "init_weights_tanh.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)

# Fitting the model to the training data and evaluating with the testing set
history_a = new_a_model.fit(X_train, Y_train,
                            batch_size=batch_size, epochs=epochs,
                            verbose=0,
                            validation_data=(X_test, Y_test))

# Plotting the loss & accuracy (with regards to the testing set)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(history_a.history['acc'])
axs[0].plot(history_a.history['val_acc'])
axs[0].set_title('Tanh Model (25 Epochs) Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(['Train', 'Test'], loc='lower right')

axs[1].plot(history_a.history['loss'])
axs[1].plot(history_a.history['val_loss'])
axs[1].set_title('Tanh Model (25 Epochs) Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(['Train', 'Test'], loc='upper right')

plt.tight_layout()
plt.show()

# Saving final weights
save_dir = "results/"
model_name = "final_weights_25epochs_tanh.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
# Evaluating model
loss_and_metrics = new_a_model.evaluate(X_test, Y_test, verbose=2)



#---------------------------------------40 EPOCHS--------------------------------------------------



#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 40


#--------------------------Implement your code here:-------------------------------------

# Initializing weights before fitting
new_a_model_40= load_model("results/init_weights_tanh.h5")

# Fitting the model to the training data and evaluating with the testing set
history_a_40 = new_a_model_40.fit(X_train, Y_train,
                            batch_size=batch_size, epochs=epochs,
                            verbose=0,
                            validation_data=(X_test, Y_test))
# Plotting the loss & accuracy (with regards to the testing set)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(history_a_40.history['acc'])
axs[0].plot(history_a_40.history['val_acc'])
axs[0].set_title('Tanh Model (40 Epochs) Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(['Train', 'Test'], loc='lower right')

axs[1].plot(history_a_40.history['loss'])
axs[1].plot(history_a_40.history['val_loss'])
axs[1].set_title('Tanh Model (40 Epochs) Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(['Train', 'Test'], loc='upper right')

plt.tight_layout()
plt.show()

# Saving the final weights
save_dir = "results/"
model_name = "final_weights_40epochs_tanh.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model_40.save(model_path)


# Evaluating model
loss_and_metrics = new_a_model_40.evaluate(X_test, Y_test, verbose=2)
#-----------------------------------------------------------------------------------------


# Build the `model_relu` again and run it with a batch size of 32 instead of 64. What are the advantages of the mini-batch vs. SGD?*


keras.backend.clear_session()

#--------------------------Impelment your code here:-------------------------------------

# Initializing weights before fitting
model_relu= load_model("results/init_weights_ReLU64.h5")

#----------------------------------------------------------------------------------------

batch_size = 32
epochs = 25

#--------------------------Impelment your code here:-------------------------------------
# Fitting the model to the training data and evaluating with the testing set
history_32 = model_relu.fit(X_train, Y_train,
                            batch_size=batch_size, epochs=epochs,
                            verbose=0,
                            validation_data=(X_test, Y_test))

# Plotting the loss & accuracy (with regards to the testing set)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(history_32.history['acc'])
axs[0].plot(history_32.history['val_acc'])
axs[0].set_title('ReLU Model (w/ batch 32) Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(['Train', 'Test'], loc='lower right')

axs[1].plot(history_32.history['loss'])
axs[1].plot(history_32.history['val_loss'])
axs[1].set_title('ReLU Model (w/ batch 32) Accuracy')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(['Train', 'Test'], loc='upper right')

plt.tight_layout()
plt.show()

loss_and_metrics = model_relu.evaluate(X_test, Y_test, verbose=2)
#----------------------------------------------------------------------------------------


# Build the `new_a_model` again and add batch normalization layers. How does it impact your results?*


keras.backend.clear_session()


#--------------------------Impelment your code here:-------------------------------------

# Setting up NN model with batch normalization layers
new_a_model = Sequential(name="new_a_model")
new_a_model.add(Flatten(input_shape=[32, 32, 1]))
new_a_model.add(Dense(300, kernel_initializer="he_normal", activation='tanh'))
new_a_model.add(BatchNormalization())
new_a_model.add(Dropout(dropout))

new_a_model.add(Dense(150, kernel_initializer="he_normal", activation='tanh'))
new_a_model.add(BatchNormalization())
new_a_model.add(Dropout(dropout))

new_a_model.add(Dense(4, kernel_initializer="he_normal", activation='softmax'))
#---------------------------------------------------------------------------------------



batch_size = 32
epochs = 25

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)
#Compile the network: 
new_a_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)

# Saving initial weights
save_dir = "results/"
model_name = "init_weights_tanh_batch.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)


#Preforming the training by using fit 
#--------------------------Impelment your code here:-------------------------------------
history = new_a_model.fit(X_train, Y_train,
                          batch_size=batch_size, epochs=epochs,
                          verbose=0,
                          validation_data=(X_test, Y_test))

# Plotting the loss & accuracy (with regards to the testing set)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(history.history['acc'])
axs[0].plot(history.history['val_acc'])
axs[0].set_title('Tanh Model (w/ batch normalization) Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(['Train', 'Test'], loc='lower right')

axs[1].plot(history.history['loss'])
axs[1].plot(history.history['val_loss'])
axs[1].set_title('Tanh Model (w/ batch normalization) Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(['Train', 'Test'], loc='upper right')

plt.tight_layout()
plt.show()

# Saving the final weights
save_dir = "results/"
model_name = "final_weights_tanh_batch.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Evaluating the model
loss_and_metrics = new_a_model.evaluate(X_test, Y_test, verbose=2)


#----------------------------------------------------------------------------------------


# ### PART 2: Convolutional Neural Network (CNN)
# ------------------------------------------------------------------------------------

# Have a look at the model below and answer the following:
# * How many layers does it have?
# * How many filter in each layer?
# * Would the number of parmaters be similar to a fully connected NN?
# * Is this specific NN performing regularization?

#number of filters
def get_net(input_shape,drop,dropRate,reg,nb_filters):
    #Defining the network architecture:
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = input_shape))
    model.add(Conv2D(filters=nb_filters[0], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_1',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=nb_filters[1], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_2',kernel_regularizer=regularizers.l2(reg)))
    if drop:    
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=nb_filters[2], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_3',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=nb_filters[3], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_4',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=nb_filters[4], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_5',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #Fully connected network tail:      
    model.add(Dense(512, activation='elu',name='FCN_1')) 
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu',name='FCN_2'))
    model.add(Dense(4, activation= 'softmax',name='FCN_3'))
    model.summary()
    return model



input_shape = (32,32,1)
learn_rate = 1e-5
decay = 1e-03
batch_size = 64
epochs = 25
drop = True
dropRate = 0.3
reg = 1e-2
nb_filters=[64, 128, 128, 256, 256]
nb_filters_2=[32, 64, 64, 128, 128]

NNet = get_net(input_shape,drop,dropRate,reg,nb_filters)


from tensorflow.keras.optimizers import *
import os
from tensorflow.keras.callbacks import *

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

#Compile the network: 
NNet.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

#Saving checkpoints during training:
Checkpath = os.getcwd()
Checkp = ModelCheckpoint(Checkpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True) #,save_freq=1)


#Preforming the training by using fit 
# IMPORTANT NOTE: This will take a few minutes!
h = NNet.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0, validation_data = (BaseX_val, BaseY_val), shuffle=True)
#NNet.save(model_fn)


# NNet.load_weights('Weights_1.h5')

results = NNet.evaluate(X_test,Y_test)
print('test loss, test acc:', results)


# ---
# <span style="color:red">***Task 2:***</span> *Number of filters* 
# 
# Rebuild the function `get_net` to have as an input argument a list of number of filters in each layers, i.e. for the CNN defined above the input should have been `[64, 128, 128, 256, 256]`. Now train the model with the number of filters reduced by half. What were the results.


#--------------------------Impelment your code here:-------------------------------------

#The function is already modified above. 
#----------------------------------------------------------------------------------------
