# Import the necessary packages:
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub
# from keras.api.models import Sequential
from tf_keras import Sequential
from keras.api.preprocessing.image import *
from keras.api.utils import get_file

# Define the URL of the pre-trained ResNetV2152 classifier in TFHub:
classifier_url = ('https://tfhub.dev/google/imagenet/'
                  'resnet_v2_152/classification/4')


# Download and instantiate the classifier hosted on TFHub:
# To download and convert such a network into a Keras model, 
# we used the convenient hub.KerasLayer class 
model = Sequential([
    hub.KerasLayer(classifier_url, input_shape=(224, 224, 3))
])


# Load the image we'll classify, convert it to a numpy array, normalize it, and wrap it into a singleton batch:
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the image file
image_path = os.path.join(script_dir, 'beetle.jpg')

image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)
image = image / 255.0
image = np.expand_dims(image, axis=0)


# Use the pre-trained model to classify the image:
predictions = model.predict(image)


# Extract the index of the most probable prediction:
predicted_index = np.argmax(predictions[0], axis=-1)


# Download the ImageNet labels into a file named ImageNetLabels.txt:
file_name = 'ImageNetLabels.txt'
file_url = ('https://storage.googleapis.com/'
            'download.tensorflow.org/data/ImageNetLabels.txt')
labels_path = get_file(file_name, file_url)


# Read the labels into a numpy array:
with open(labels_path) as f:
    imagenet_labels = np.array(f.read().splitlines())


# Extract the name of the class corresponding to the index of the most probable prediction:
predicted_class = imagenet_labels[predicted_index]
print(predicted_class)


# Plot the original image with its most probable label:
plt.figure()
plt.title(f'Label: {predicted_class}.')
original = load_img(image_path)
original = img_to_array(original)
plt.imshow(original / 255.0)
plt.show()
