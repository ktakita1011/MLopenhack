from flask import Flask, jsonify, abort, make_response
# import json
from pathlib import Path

from PIL import Image
from PIL import ImageOps
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

MODEL_NAME  ="AdventureWorksModel"
IMG_SIZE = 128
LR = 1e-3
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

model.load(MODEL_NAME)

api = Flask(__name__)

@api.route('/predict/<string:imagepath>', methods=['GET'])
def predict(imagepath):
   try:
       x = []
       im = Image.open("C:\\Users\\kenakamu\\Downloads\\gear_images\\gear_images_128\\insulated_jackets\\105017.jpeg")
       arr = np.array(im)
       x.append(arr)
       result = model.predict(x)
   except:
       abort(404)

   result = {
       "result":True,
       "data":{
           "result":str(result[0])
           }
       }

   return make_response(jsonify(result))
   # Unicodeにしたくない場合は↓
   # return make_response(json.dumps(result, ensure_ascii=False))


@api.route('/getUser/<string:userId>', methods=['GET'])
def get_user(userId):
   try:
       user = userId
   except User.DoesNotExist:
       abort(404)

   result = {
       "result":True,
       "data":{
           "userId":userId
           }
       }

   return make_response(jsonify(result))
   # Unicodeにしたくない場合は↓
   # return make_response(json.dumps(result, ensure_ascii=False))

@api.errorhandler(404)
def not_found(error):
   return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
   api.run(host='0.0.0.0', port=3000)