# Import the necessary packages:
import os
import tarfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.api.utils import get_file


# Define the URL and destination of the CINIC-10 dataset, an alternative to the famous CIFAR-10 dataset:
DATASET_URL = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz?sequence=4&isAllowed=y'
DATA_NAME = 'cinic10'
FILE_EXTENSION = 'tar.gz'
FILE_NAME = '.'.join([DATA_NAME, FILE_EXTENSION])


# Download and decompress the data. By default, it will be stored in ~/.keras/datasets/<FILE_NAME>:
downloaded_file_location = get_file(origin=DATASET_URL, fname=FILE_NAME, extract=False)

# Build the path to the data directory based on the location of the downloaded file.
data_directory, _ = downloaded_file_location.rsplit(os.path.sep, maxsplit=1)
data_directory = os.path.sep.join([data_directory, DATA_NAME])

# Only extract the data if it hasn't been extracted already
if not os.path.exists(data_directory):
    tar = tarfile.open(downloaded_file_location)
    tar.extractall(data_directory)

print(f'Data downloaded to {data_directory}')


# Create a dataset of image paths using a glob-like pattern:
data_pattern = os.path.sep.join([data_directory, '*/*/*.png'])
image_dataset = tf.data.Dataset.list_files(data_pattern)


# Take a single path from the dataset and use it to read the corresponding image:
for file_path in image_dataset.take(1):
    sample_path = file_path.numpy()
    print(f'Sample image path: {sample_path}')

sample_image = tf.io.read_file(sample_path)
print(f'Image type: {type(sample_image)}')


# Even though the image is now in memory, we must convert it into a format a neural network can work with. 
# For this, we must decode it from its PNG format into a NumPy array, as follows:
sample_image = tf.image.decode_png(sample_image, channels=3)
sample_image = sample_image.numpy()


# Display the image using matplotlib:
plt.imshow(sample_image / 255.0)


# Take the first 10 elements of image_dataset, decode and normalize them, and then display them using matplotlib:
plt.figure(figsize=(5, 5))
for index, image_path in enumerate(image_dataset.take(10), start=1):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, np.float32)

    ax = plt.subplot(5, 5, index)
    plt.imshow(image)
    plt.axis('off')

plt.show()
plt.close()
