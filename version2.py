# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
#####################################################################################################################################
# input image dimensions
img_rows, img_cols = 32, 32

# number of channels
img_channels = 1

#####################################################################################################################################
#path1 = './pic/pic/'
path1 = './project/lp/pictures/'
listing = os.listdir(path1)
listing = listing[1:]
path2 = './project/picst/'
num_samples=size(listing)
#immatrix1 = []
for file in listing:
    path = str(path1)+file
        #im1 = array(Image.open(path))
        im = Image.open(path)
            img = im.resize((img_rows,img_cols))
                img.save('./project/picst' +'/' +  file, "JPEG")

imlist = os.listdir(path2)
immatrix = array([array(Image.open('./project/picst'+ '/' + im2)).flatten()for im2 in imlist],'f')

label=np.ones((num_samples,),dtype = int)
label[0:3778]=0
label[3778:]=1

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 3
# number of epochs to train
nb_epoch = 9


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#%%
(X, y) = (train_data[0],train_data[1])

# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)


X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#####################################################################################################################################
model = Sequential()

model.add(Convolution2D(20, nb_conv, nb_conv,border_mode='same',input_shape=(1, img_rows, img_cols)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides = (1,1)))

model.add(Convolution2D(50, nb_conv, nb_conv,border_mode='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), strides = (1,1)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

model.add(Dense(nb_classes))
model.add(Activation("softmax"))

SGD = SGD(lr=0.01, momentum=0.9, decay= 0.0005 , nesterov=False)
model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1,shuffle=True,  validation_data=(X_test, Y_test))
#####################################################################################################################################
score = model.evaluate(X_test, Y_test,batch_size=32, verbose=0)
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
xc=range(nb_epoch)

plt.figure(1,figsize=(5,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.savefig('project4.png')
plt.show()
#####################################################################################################################################
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# saving weights
fname = "version2.hdf5"
model.save_weights(fname,overwrite=True)

  
