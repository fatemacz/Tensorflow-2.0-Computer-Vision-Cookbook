# Import the necessary packages:
import os
import pathlib

import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.api.models import Model
from keras.src.layers import *
from keras.src.losses.losses import CategoricalCrossentropy
import kagglehub


# Define a list with the three classes, and also an alias to tf.data.experimental.AUTOTUNE, which we'll use later:
# The values in CLASSES match the names of the directories that contain the images for each class.
CLASSES = ['rock', 'paper', 'scissors']
AUTOTUNE = tf.data.experimental.AUTOTUNE


# Define a function to load an image and its label, given its file path:
# Notice that we are one-hot encoding by comparing the name of the folder 
# that contains the image (extracted from image_path) with the CLASSES list.
def load_image_and_label(image_path, target_size=(32, 32)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.convert_image_dtype(image, np.float32)
    image = tf.image.resize(image, target_size)

    label = tf.strings.split(image_path, os.path.sep)[-2]
    label = (label == CLASSES)  # One-hot encode.
    label = tf.dtypes.cast(label, tf.float32)

    return image, label


# Define a function to build the network architecture. 
# In this case, it's a very simple and shallow one, which is enough for the problem we are solving:
def build_network():
    input_layer = Input(shape=(32, 32, 1))
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same',
               strides=(1, 1))(input_layer)
    x = ReLU()(x)
    x = Dropout(rate=0.5)(x)

    x = Flatten()(x)
    x = Dense(units=3)(x)
    output = Softmax()(x)

    model = Model(inputs=input_layer, outputs=output)
    return model


# Define a function to, given a path to a dataset, 
# return a tf.data.Dataset instance of images and labels, in batches and optionally shuffled:
def prepare_dataset(dataset_path,
                    buffer_size,
                    batch_size,
                    shuffle=True):
    dataset = (tf.data.Dataset
               .from_tensor_slices(dataset_path)
               .map(load_image_and_label,
                    num_parallel_calls=AUTOTUNE))

    if shuffle:
        dataset.shuffle(buffer_size=buffer_size)

    dataset = (dataset
               .batch(batch_size=batch_size)
               .prefetch(buffer_size=buffer_size))

    return dataset


# Load the image paths into a list:

# # download zip from https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors
# # and extract to <Path to User>/.keras/datasets/rockpaperscissors
# # *******************************************************************************
# file_pattern = (pathlib.Path.home() / '.keras' / 'datasets' /
#                'rockpaperscissors' / 'rps-cv-images' / '*' /
#                '*.png')
# file_pattern = str(file_pattern)
# # *******************************************************************************
# # <OR>
# # *******************************************************************************
# # Download with code below
download_path = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
print("Path to dataset files:", download_path)
file_pattern = os.path.join(os.path.abspath(download_path), 'rps-cv-images', '*', '*.png')
# # *******************************************************************************

dataset_paths = [*glob.glob(file_pattern)]


# Create train, test, and validation subsets of image paths:
train_paths, test_paths = train_test_split(dataset_paths,
                                           test_size=0.2,
                                           random_state=999)
train_paths, val_paths = train_test_split(train_paths,
                                          test_size=0.2,
                                          random_state=999)


# Prepare the training, test, and validation datasets:
BATCH_SIZE = 1024
BUFFER_SIZE = 1024

train_dataset = prepare_dataset(train_paths,
                                buffer_size=BUFFER_SIZE,
                                batch_size=BATCH_SIZE)
validation_dataset = prepare_dataset(val_paths,
                                     buffer_size=BUFFER_SIZE,
                                     batch_size=BATCH_SIZE,
                                     shuffle=False)
test_dataset = prepare_dataset(test_paths,
                               buffer_size=BUFFER_SIZE,
                               batch_size=BATCH_SIZE,
                               shuffle=False)


# Instantiate and compile the model:
model = build_network()
model.compile(loss=CategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])


# Fit the model for 250 epochs:
EPOCHS = 250
model.fit(train_dataset,
          validation_data=validation_dataset,
          epochs=EPOCHS)


# Evaluate the model on the test set:
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Loss: {test_loss}, accuracy: {test_accuracy}')
