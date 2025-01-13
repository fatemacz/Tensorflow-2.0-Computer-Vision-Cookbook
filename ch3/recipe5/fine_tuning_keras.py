# Import the necessary dependencies:
import os
import pathlib
from glob import glob

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.api.models import Model
from keras.api.applications.vgg16 import VGG16
from keras.src.layers import *
from keras.api.optimizers import *
from keras.api.preprocessing.image import *
from keras.src.legacy.preprocessing.image import ImageDataGenerator


# Set the random seed:
SEED = 999


# Define a function that will build a new network from a pre-trained model, 
# where the top fully connected layers will be brand new and adapted to the problem at hand:
def build_network(base_model, classes):
    x = Flatten()(base_model.output)
    x = Dense(units=256)(x)
    x = ReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(rate=0.5)(x)

    x = Dense(units=classes)(x)
    output = Softmax()(x)

    return output


# Define a function that will load the images and labels in the dataset as NumPy arrays:
def load_images_and_labels(image_paths,
                           target_size=(256, 256)):
    images = []
    labels = []

    for image_path in image_paths:
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)

        label = image_path.split(os.path.sep)[-2]

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)


# Load the image paths and extract the set of classes from them:
dataset_path = (pathlib.Path.home() / '.keras' / 'datasets' /
                'flowers17')
files_pattern = (dataset_path / 'images' / '*' / '*.jpg')
image_paths = [*glob(str(files_pattern))]
CLASSES = {p.split(os.path.sep)[-2] for p in image_paths}


# Load the images and normalize them, one-hot encode the labels with LabelBinarizer(), 
# and split the data into subsets for training (80%) and testing (20%):
X, y = load_images_and_labels(image_paths)
X = X.astype('float') / 255.0
y = LabelBinarizer().fit_transform(y)

(X_train, X_test,
 y_train, y_test) = train_test_split(X, y,
                                     test_size=0.2,
                                     random_state=SEED)


# Instantiate a pre-trained VGG16, without the top layers. Specify an input shape of 256x256x3:
base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_tensor=Input(shape=(256, 256, 3)))


# Freeze all the layers in the base model. 
# We are doing this because we don't want to re-train them, but use their existing knowledge:
for layer in base_model.layers:
    layer.trainable = False


# Build the full network with a new set of layers on top using build_network() (defined in Step 3):
model = build_network(base_model, len(CLASSES))
model = Model(base_model.input, model)


# Define the batch size and a set of augmentations to be applied through ImageDataGenerator():
BATCH_SIZE = 64
augmenter = ImageDataGenerator(rotation_range=30,
                               horizontal_flip=True,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.2,
                               zoom_range=0.2,
                               fill_mode='nearest')
train_generator = augmenter.flow(X_train, y_train, BATCH_SIZE)


# Warm up the network. 
# This means we'll only train the new layers (the rest are frozen) for 20 epochs, using RMSProp with a learning rate of 0.001. 
# Finally, we'll evaluate the network on the test set:
WARMING_EPOCHS = 20
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=1e-3),
              metrics=['accuracy'])
history = model.fit(train_generator,
                    steps_per_epoch=len(X_train) // BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    epochs=WARMING_EPOCHS)
result = model.evaluate(X_test, y_test)
print(f'Test accuracy: {result[1]}')


# Now that the network has been warmed up, 
# we'll fine-tune the final layers of the base model, specifically from the 16th onward (remember, zero-indexing), 
# along with the fully connected layers, for 50 epochs, using SGD with a learning rate of 0.001:
for layer in base_model.layers[15:]:
    layer.trainable = True

EPOCHS = 50
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(learning_rate=1e-3),
              metrics=['accuracy'])
history = model.fit(train_generator,
                    steps_per_epoch=len(X_train) // BATCH_SIZE,
                    validation_data=(X_test, y_test),
                    epochs=EPOCHS)
result = model.evaluate(X_test, y_test)
print(f'Test accuracy: {result[1]}')
