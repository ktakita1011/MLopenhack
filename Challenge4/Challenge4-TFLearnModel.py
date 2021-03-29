
# coding: utf-8

# In[1]:


# Challenge 4: Deep Learning

import tensorflow as tf

# Clears the default graph stack and resets the global default graph.
tf.reset_default_graph() 


# In[2]:


from pathlib import Path

from PIL import Image
from PIL import ImageOps
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils  import to_categorical
import tensorflow as tf
data_path = './gear_images_128'
category = 0 # Category label
labels = [] # Label strings
x = [] # Image data
y = [] # Labels for each image

def prepare_dataset(file):    
    try:
        im = Image.open(file)
        arr = np.array(im)
        x.append(arr)
        y.append(to_categorical(category, num_classes=12)) # if category is 1, then it returns [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    except:
        print("cannot add to dataset:", file)

# Loop all data and create dataset
for dir in Path(data_path).iterdir():
    labels.append(str(dir).split('/')[-1])
    for file in dir.iterdir():
        prepare_dataset(file)
    print(dir, ':', category)
    category += 1

x = np.asarray(x,dtype=float) # Convert image list into array of float.
x = x/255 # Make value between 0 to 1
y = np.asarray(y,dtype=float)

# Split data into 70% train and 30% test. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[3]:


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

IMG_SIZE = 128
LR = 1e-3

# Input data shape is 4-D Tensor. [batch, height, width. channel]
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.5)
convnet = fully_connected(convnet, 12, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


# In[4]:


MODEL_NAME  ="AdventureWorksModel-RegressionFirst"
model.fit({'input': x_train}, {'targets': y_train}, n_epoch=10, validation_set=({'input': x_test}, {'targets': y_test}), snapshot_step=10,)


# In[18]:


model.save("MODEL_NAME")


# In[19]:


model.load("AdventureWorksModel.tflearn")


# In[20]:


model.evaluate(x_test, y_test)


# In[29]:


y_predict = model.predict(x_test)


# In[48]:


y_test.shape
print(y_test)
y_test_np = np.argmax(y_test,1)
print(y_test_np)
print(y_test_np.shape)


# In[47]:


y_predict.shape
print(y_predict)
y_pre_np = np.argmax(y_predict,1)
print(y_pre_np)
print(y_pre_np.shape)


# In[49]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_np,y_pre_np)


# In[50]:


from sklearn.metrics import classification_report

print(classification_report(y_test_np, y_pre_np))


# In[58]:


predict_classes = model.predict_classes(x_test, batch_size=1)
prediction = []
prediction.append(y_predict)
labels.append(y_test)
confusionMat=tf.confusion_matrix(y_test,y_predict,2)

