import os
import kagglehub
from tqdm import tqdm
import pathlib
from glob import glob

# path = 'C:\\Users\\AZ872\\.keras\\datasets\\caltech-101\\101_ObjectCategories\\accordion\\image_0001.jpg'
# split_path = path.split(os.path.sep)
# classes = split_path[-2]
# print(classes)

# abs_path_name = os.path.abspath(__file__)
# script_dir = os.path.dirname(abs_path_name)
# # Construct the path to the image file
# image_path = os.path.join(script_dir, 'test.png')

# download_path = kagglehub.dataset_download("peterjun/car196")
# print("Path to dataset files:", download_path)

# dataset_path = (pathlib.Path.home() / '.keras' / 'datasets' /
#                 'flowers17')
# files_pattern = (dataset_path / 'images' / '*' / '*.jpg')
# images_path = [*glob(str(files_pattern))]

# labels = []
# for index in tqdm(range(len(images_path))):
#     image_path = images_path[index]
#     label = image_path.split(os.path.sep)[-2]
#     labels.append(label)

# print(labels)

import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.debugging.set_log_device_placement(True)

try:
    # Specify an invalid GPU device
    with tf.device('/device:GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
except RuntimeError as e:
    print(e)