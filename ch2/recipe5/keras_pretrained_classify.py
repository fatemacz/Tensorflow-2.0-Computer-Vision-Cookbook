# Import the necessary packages:
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.api.applications import imagenet_utils
from keras.src.applications.inception_v3 import *
from keras.api.preprocessing.image import *


# Instantiate an InceptionV3 network pre-trained on ImageNet:
model = InceptionV3(weights='imagenet')


# Load the image to classify. InceptionV3 takes a 299x299x3 image, so we must resize it accordingly:
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the image file
image_path = os.path.join(script_dir, 'dog.jpg')
image = load_img(image_path, target_size=(299, 299))


# Convert the image to a numpy array, and wrap it into a singleton batch:
image = img_to_array(image)
image = np.expand_dims(image, axis=0)


# Pre process the image the same way InceptionV3 does:
image = preprocess_input(image)


# Use the model to make predictions on the image, and then decode the predictions to a matrix:
predictions = model.predict(image)
prediction_matrix = (imagenet_utils
                     .decode_predictions(predictions))


# Examine the top 5 predictions along with their probability:
for i in range(5):
    imagenet_id, label, probability = prediction_matrix[0][i]
    print(f'{i + 1}. {label}: {probability * 100:.3f}%')


# Plot the original image with its most probable label:
_, label, _ = prediction_matrix[0][0]
plt.figure()
plt.title(f'Label: {label}.')
original = load_img(image_path)
original = img_to_array(original)
plt.imshow(original / 255.0)
plt.show()
