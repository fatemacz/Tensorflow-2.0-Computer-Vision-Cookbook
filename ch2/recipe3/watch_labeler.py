# Import the necessary packages:
import os
import pathlib
from csv import DictReader

import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.src.layers import *
from keras.api.models import Model
from keras.api.preprocessing.image import *
import kagglehub


# Define a function to build the network architecture. 
# First, implement the convolutional blocks:
def build_network(width, height, depth, classes):
    input_layer = Input(shape=(width, height, depth))

    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same')(input_layer)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               padding='same')(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.25)(x)

# add the fully convolutional layers:
    x = Flatten()(x)
    x = Dense(units=512)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(units=classes)(x)
    output = Activation('sigmoid')(x)

    return Model(input_layer, output)


# Define a function to load all images and labels (gender and usage), 
# given a list of image paths and a dictionary of metadata associated with each of them:
def load_images_and_labels(image_paths, styles, target_size):
    images = []
    labels = []

    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        image_id = image_path.split(os.path.sep)[-1][:-4]

        image_style = styles[image_id]
        label = (image_style['gender'], image_style['usage'])

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


# Set the random seed to guarantee reproducibility:
SEED = 999
np.random.seed(SEED)

# # *******************************************************************************
# # download zip from https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors
# # and extract to <Path to User>/.keras/datasets/rockpaperscissors

# base_path = (pathlib.Path.home() / '.keras' / 'datasets' /
#              'fashion-product-images-small')
# # Define the paths to the images and the styles.csv metadata file:
# styles_path = str(base_path / 'styles.csv')
# images_path_pattern = str(base_path / 'images/*.jpg')
# # *******************************************************************************
# # <OR>
# # *******************************************************************************
# # Download with code below

base_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
print("Path to dataset files:", base_path)
# Define the paths to the images and the styles.csv metadata file:
styles_path = os.path.join(os.path.abspath(base_path), 'styles.csv')
images_path_pattern = os.path.join(os.path.abspath(base_path), 'images', '*.jpg')
# # *******************************************************************************

image_paths = glob.glob(images_path_pattern)


# Keep only the Watches images for Casual, Smart Casual, and Formal usage, suited to Men and Women:
with open(styles_path, 'r') as f:
    dict_reader = DictReader(f)
    STYLES = [*dict_reader]

    article_type = 'Watches'
    genders = {'Men', 'Women'}
    usages = {'Casual', 'Smart Casual', 'Formal'}
    STYLES = {style['id']: style
              for style in STYLES
              if (style['articleType'] == article_type and
                  style['gender'] in genders and
                  style['usage'] in usages)}

image_paths = [*filter(lambda p: p.split(os.path.sep)[-1][:-4]
                                 in STYLES.keys(),
                       image_paths)]


# Load the images and labels, resizing the images into a 64x64x3 shape:
X, y = load_images_and_labels(image_paths, STYLES, (64, 64))


# Normalize the images and multi-hot encode the labels:
X = X.astype('float') / 255.0
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)


# Create the train, validation, and test splits:
(X_train, X_test,
 y_train, y_test) = train_test_split(X, y,
                                     stratify=y,
                                     test_size=0.2,
                                     random_state=SEED)
(X_train, X_valid,
 y_train, y_valid) = train_test_split(X_train, y_train,
                                      stratify=y_train,
                                      test_size=0.2,
                                      random_state=SEED)


# Build and compile the network:
model = build_network(width=64,
                      height=64,
                      depth=3,
                      classes=len(mlb.classes_))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# Train the model for 20 epochs, in batches of 64 images at a time:
BATCH_SIZE = 64
EPOCHS = 20
model.fit(X_train, y_train,
          validation_data=(X_valid, y_valid),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)


# Evaluate the model on the test set:
result = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print(f'Test accuracy: {result[1]}')


# Use the model to make predictions on a test image, displaying the probability of each label:
test_image = np.expand_dims(X_test[0], axis=0)
probabilities = model.predict(test_image)[0]

for label, p in zip(mlb.classes_, probabilities):
    print(f'{label}: {p * 100:.2f}%')


# Compare the ground truth labels with the network's prediction:
ground_truth_labels = np.expand_dims(y_test[0], axis=0)
ground_truth_labels = mlb.inverse_transform(ground_truth_labels)
print(f'Ground truth labels: {ground_truth_labels}')
