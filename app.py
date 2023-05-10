from flask import Flask,render_template,request
from PIL import Image
import io


#from this ....

import tensorflow as tf

from keras import optimizers
from keras.utils import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.applications import ResNet50, DenseNet201
from keras.applications import resnet, densenet

import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import pandas as pd


train_path = "data/train"
valid_path = "data/valid"
test_path = "data/test"

image_shape = (305,430,3)
N_CLASSES = 4
BATCH_SIZE = 32


train_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
train_generator = train_datagen.flow_from_directory(train_path,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (305,430),
                                                   class_mode = 'categorical')

valid_datagen = ImageDataGenerator(dtype='float32', rescale= 1./255.)
valid_generator = valid_datagen.flow_from_directory(valid_path,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (305,430),
                                                   class_mode = 'categorical')

test_datagen = ImageDataGenerator(dtype='float32', rescale = 1.0/255.0)
test_generator = test_datagen.flow_from_directory(test_path,
                                                   batch_size = BATCH_SIZE,
                                                   target_size = (305,430),
                                                   class_mode = 'categorical')

from tensorflow.keras.models import load_model


loaded_model = load_model('models/finalmodel-ResNet50.hdf5')




app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/aboutUs')
def aboutUs():
    return render_template('aboutUs.html')


@app.route('/upload' , methods = ['POST'])
def upload():
    img_path = request.files['file']
    image = Image.open(io.BytesIO(img_path.read()))
    class_names=list(test_generator.class_indices.keys())

    # Save the image to a temporary file
    temp_file = 'temp\image.png'
    image.save(temp_file)

    img = tf.keras.utils.load_img(temp_file, target_size=(460, 460))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    prediction = loaded_model.predict(img_array)

    predicted_class_index = np.argmax(prediction)
    confidence = 100 * np.max(prediction)

    # Format the string with the predicted class and confidence
    return "<br><br><br><br><br><br><br><br><h1>                                This image most likely belongs to {} with a {:.2f} percent confidence." \
            .format(class_names[predicted_class_index], confidence) + "<h1>"

    



if __name__ == '__main__':
    app.run(debug=True)

