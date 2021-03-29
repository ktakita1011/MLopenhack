
# coding: utf-8

# In[4]:


import numpy as np
from PIL import Image, ImageOps
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard

#from keras.datasets import cifar10
from keras.utils import np_utils

from keras.preprocessing.text import Tokenizer

from keras.models import load_model

import matplotlib

import os
import glob


# In[4]:


from keras.models import Sequential


# In[5]:


files = glob.glob('./converted/*/*.jpeg')

images = []
targets = []

for f in files:
    # file data to array
    arr = np.array(Image.open(f))#.flatten()
    images.append(arr)
    
    target = os.path.basename(os.path.dirname(f))
    targets.append(target)

data_train, data_test, label_train, label_test = train_test_split(images, targets, test_size=0.3)


# In[33]:


data_train = np.array(data_train)
data_test = np.array(data_test)
#label_train = np.array(label_train)

label = list(set(label_train))
label
label_train2 = []

for i in label_train:
    label_train2.append(label.index(i))

label_train3 = np_utils.to_categorical(label_train2 , 12)


# In[34]:


data_train = data_train.astype('float32')/255
data_test = data_test.astype('float32')/255


# In[45]:


classes = 12

model = Sequential()

model.add(Conv2D(32,4,input_shape=(128,128,3)))
model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(128,3))
model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(classes))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])


# In[51]:


batch_size = 100
nb_epoch = 20
history = model.fit(data_train, label_train3, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.1)


# In[41]:


data_test[0][0]


# In[5]:


model = load_model('model.h5')


# In[56]:


history.history


# In[6]:


print(model.summary())


# In[7]:


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

