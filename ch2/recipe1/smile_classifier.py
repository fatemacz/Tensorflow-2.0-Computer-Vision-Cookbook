# Import the necessary packages:
import os
import pathlib
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.api.models import Model
from keras.src.layers import *
from keras.api.preprocessing.image import *


# Define a function to load the images and labels from a list of file paths:
# Notice that we are loading the images in grayscale, 
# and we're encoding the labels by checking whether the word positive is in the file path of the image.
def load_images_and_labels(image_paths):
    images = []
    labels = []

    for image_path in image_paths:
        image = load_img(image_path, target_size=(32, 32),
                         color_mode='grayscale')
        image = img_to_array(image)

        label = image_path.split(os.path.sep)[-2]
        label = 'positive' in label
        label = float(label)

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


# Define a function to build the neural network. 
# This model's structure is based on LeNet (you can find a link to LeNet's paper in the See also section):
# Because this is a binary classification problem, a single Sigmoid-activated neuron is enough in the output layer.
def build_network():
    input_layer = Input(shape=(32, 32, 1))
    x = Conv2D(filters=20,
               kernel_size=(5, 5),
               padding='same',
               strides=(1, 1))(input_layer)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2),
                     strides=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filters=50,
               kernel_size=(5, 5),
               padding='same',
               strides=(1, 1))(x)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2),
                     strides=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(units=500)(x)
    x = ELU()(x)
    x = Dropout(0.4)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)
    return model


# Load the image paths into a list:
# under <Path to User>/.keras/datasets/
# git clone https://github.com/hromi/SMILEsmileD.git 
files_pattern = (pathlib.Path.home() / '.keras' / 'datasets' /
                 'SMILEsmileD' / 'SMILEs' / '*' / '*' /
                 '*.jpg')
files_pattern = str(files_pattern)
dataset_paths = [*glob.glob(files_pattern)]


# Use the load_images_and_labels() function defined previously to load the dataset into memory:
X, y = load_images_and_labels(dataset_paths)


# Normalize the images and compute the number of positive, negative, and total examples in the dataset:
X /= 255.0
total = len(y)
total_positive = np.sum(y)
total_negative = total - total_positive
print(f'Total images: {total}')
print(f'Smile images: {total_positive}')
print(f'Non-smile images: {total_negative}')


# Create train, test, and validation subsets of the data:
(X_train, X_test,
 y_train, y_test) = train_test_split(X, y,
                                     test_size=0.2,
                                     stratify=y,
                                     random_state=999)
(X_train, X_val,
 y_train, y_val) = train_test_split(X_train, y_train,
                                    test_size=0.2,
                                    stratify=y_train,
                                    random_state=999)


# Instantiate the model and compile it:
model = build_network()
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# Train the model. 
# Because the dataset is unbalanced, 
# we are assigning weights to each class proportional to the number of positive and negative images in the dataset:
BATCH_SIZE = 32
EPOCHS = 20
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          class_weight={
              1.0: total / total_positive,
              0.0: total / total_negative
          })
# To account for the imbalance in the dataset (out of the 13,165 images, 
# only 3,690 contain smiling people, while the remaining 9,475 do not), 
# we passed a class_weight dictionary where we assigned a weight conversely proportional to the number of instances of each class 
# in the dataset, effectively forcing the model to pay more attention to the 1.0 class, which corresponds to smile.


# Evaluate the model on the test set:
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {test_loss}, accuracy: {test_accuracy}')
