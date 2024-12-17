# Import the necessary packages:
import glob
import os
import tarfile

import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.api.preprocessing.image import load_img, img_to_array
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

# Load all image paths and print the number of images found:
data_pattern = os.path.sep.join([data_directory, '*/*/*.png'])

image_paths = list(glob.glob(data_pattern))
print(f'Sample image path: {image_paths[0]}')

# Load a single image from the dataset and print its metadata:
sample_image = load_img(image_paths[0])
print(f'Image type: {type(sample_image)}')
print(f'Image format: {sample_image.format}')
print(f'Image mode: {sample_image.mode}')
print(f'Image size: {sample_image.size}')

# Convert an image into a NumPy array:
sample_image_array = img_to_array(sample_image)
print(f'Image type: {type(sample_image_array)}')
print(f'Image array shape: {sample_image_array.shape}')


# Display an image using matplotlib:
plt.imshow(sample_image_array / 255.0)

# Load a batch of images using ImageDataGenerator. 
# As in the previous step, each image will be rescaled to the range [0, 1]:
scale_factor = 1.0 / 255.0
image_generator = ImageDataGenerator(horizontal_flip=True, rescale=scale_factor)


# Using image_generator, we'll pick and display a random batch of 10 images directly from the directory they are stored in:
iterator = (
    image_generator.flow_from_directory(
        directory=data_directory,
        batch_size=10
    )
)
for batch, _ in iterator:
    plt.figure(figsize=(5, 5))
    for index, image in enumerate(batch, start=1):
        ax = plt.subplot(5, 5, index)
        plt.imshow(image)
        plt.axis('off')

    plt.show()
    break
