# Import the necessary packages:
import glob
import os
import pathlib
import kagglehub
import h5py
import numpy as np
import sklearn.utils as skutils
from sklearn.preprocessing import LabelEncoder
from keras.api.applications import imagenet_utils
from keras.api.applications.vgg16 import VGG16
from keras.api.preprocessing.image import *
from tqdm import tqdm


# Define the FeatureExtractor class and its constructor:
class FeatureExtractor(object):
    def __init__(self,
                 model,
                 input_size,
                 label_encoder,
                 num_instances,
                 feature_size,
                 output_path,
                 features_key='features',
                 buffer_size=1000):
        # We need to make sure the output path can be written:
        if os.path.exists(output_path):
            error_msg = (f'{output_path} already exists. '
                         f'Please delete it and try again.')
            raise FileExistsError(error_msg)
        
        # Now, let's store the input parameter as object members:
        self.model = model
        self.input_size = input_size
        self.le = label_encoder
        self.feature_size = feature_size

        # self.buffer will contain a buffer of both instances and labels, 
        # while self.current_index will point to the next free location within the datasets in the inner HDF5 database. 
        # We'll create this now:
        self.db = h5py.File(output_path, 'w')
        self.features = self.db.create_dataset(features_key,
                                               (num_instances,
                                                feature_size),
                                               dtype='float')
        self.labels = self.db.create_dataset('labels',
                                             (num_instances,),
                                             dtype='int')

        self.buffer_size = buffer_size
        self.buffer = {'features': [], 'labels': []}
        self.current_index = 0


    # Define a method that will extract features and labels from a list of image paths and store them in the HDF5 database:
    def extract_features(self,
                         image_paths,
                         labels,
                         batch_size=64,
                         shuffle=True):
        if shuffle:
            image_paths, labels = skutils.shuffle(image_paths,
                                                  labels)

        encoded_labels = self.le.fit_transform(labels)

        self._store_class_labels(self.le.classes_)
        
        # After shuffling the image paths and their labels, as well as encoding and storing the latter, 
        # we'll iterate over batches of images, passing them through the pre-trained network. 
        # Once we've done this, we'll save the resulting features into the HDF5 database 
        # (the helper methods we've used here will be defined shortly):
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i: i + batch_size]
            batch_labels = encoded_labels[i:i + batch_size]
            batch_images = []

            for image_path in batch_paths:
                image = load_img(image_path,
                                 target_size=self.input_size)
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = imagenet_utils.preprocess_input(image)

                batch_images.append(image)

            batch_images = np.vstack(batch_images)
            feats = self.model.predict(batch_images,
                                       batch_size=batch_size)

            new_shape = (feats.shape[0], self.feature_size)
            feats = feats.reshape(new_shape)
            self._add(feats, batch_labels)

        self._close()


    # Define a private method that will add features and labels to the corresponding datasets:
    def _add(self, rows, labels):
        self.buffer['features'].extend(rows)
        self.buffer['labels'].extend(labels)

        if len(self.buffer['features']) >= self.buffer_size:
            self._flush()


    # Define a private method that will flush the buffers to disk:
    def _flush(self):
        next_index = (self.current_index +
                      len(self.buffer['features']))
        buffer_slice = slice(self.current_index, next_index)
        self.features[buffer_slice] = self.buffer['features']
        self.labels[buffer_slice] = self.buffer['labels']
        self.current_index = next_index
        self.buffer = {'features': [], 'labels': []}


    # Define a private method that will store the class labels in the HDF5 database:
    def _store_class_labels(self, class_labels):
        data_type = h5py.string_dtype(encoding='utf-8')
        label_ds = self.db.create_dataset('label_names',
                                          (len(class_labels),),
                                          dtype=data_type)
        label_ds[:] = [str(label) for label in class_labels]


    # Define a private method that will close the HDF5 dataset:
    def _close(self):
        if len(self.buffer['features']) > 0:
            self._flush()

        self.db.close()


if __name__ == '__main__':
    # Download dataset and load the paths to the images in the dataset:
    download_path = kagglehub.dataset_download("peterjun/car196")
    print("Path to dataset files:", download_path)
    # download_path = r'C:\Users\AZ872\.cache\kagglehub\datasets\peterjun\car196\versions\2'
    files_pattern = os.path.join(os.path.abspath(download_path), 'car196', '*', '*', '*.jpg')

    # files_pattern = (pathlib.Path.home() / '.keras' / 'datasets' /
    #                  'car_ims' / '*.jpg')
    # files_pattern = str(files_pattern)

    input_paths = [*glob.glob(files_pattern)]
    # output_path = (pathlib.Path.home() / '.keras' / 'datasets' /
    #                'car_ims_rotated')


    # Create the output directory. 
    # We'll create a dataset of rotated car images 
    # so that a potential classifier can learn how to correctly revert the photos back to their original orientation, 
    # by correctly predicting the rotation angle:
    output_path = os.path.join(os.path.abspath(download_path), 'car_ims_rotated')

    if not os.path.exists(str(output_path)):
        os.mkdir(str(output_path))


    # Create a copy of the dataset with random rotations performed on the images:
    labels = []
    output_paths = []
    for index in tqdm(range(len(input_paths))):
        image_path = input_paths[index]
        image = load_img(image_path)
        rotation_angle = np.random.choice([0, 90, 180, 270])

        rotated_image = image.rotate(rotation_angle)
        rotated_image_path = os.path.join(output_path, f'{index}.jpg')
        rotated_image.save(rotated_image_path, 'JPEG')

        output_paths.append(rotated_image_path)
        labels.append(rotation_angle)

        image.close()
        rotated_image.close()


    # Instantiate FeatureExtractor while using a pre-trained VGG16 network to extract features from the images in the dataset:
    # features_path = str(output_path / 'features.hdf5')
    features_path = os.path.join(output_path, 'features.hdf5')
    model = VGG16(weights='imagenet', include_top=False)
    fe = FeatureExtractor(model=model,
                        input_size=(224, 224, 3),
                        label_encoder=LabelEncoder(),
                        num_instances=len(input_paths),
                        feature_size=512 * 7 * 7,
                        output_path=features_path)


    # Extract the features and labels:
    fe.extract_features(image_paths=output_paths,
                        labels=labels)
