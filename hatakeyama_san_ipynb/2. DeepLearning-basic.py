
# coding: utf-8

# # Deep Learning 基礎
# ## Keras で MNIST 実行
# 
# ## 必要なもの
# 
# 
# 
# 参考: https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py

# In[1]:


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# ## ハイパーパラメーター設定

# In[2]:


batch_size = 128
num_classes = 10
epochs = 20


# ## データのロード

# In[3]:


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[4]:


# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# ## Neural Network 作成

# In[5]:


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


# # トレーニング実行

# In[6]:


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# # モデルの評価

# In[7]:


from sklearn.metrics import confusion_matrix
import numpy as np

predict_classes = model.predict_classes(x_test, batch_size=1)
true_classes = np.argmax(y_test,1)

confusion_matrix(true_classes, predict_classes)


# In[8]:


from sklearn.metrics import classification_report

print(classification_report(true_classes, predict_classes))

