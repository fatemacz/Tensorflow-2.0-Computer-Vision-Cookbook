# Import the necessary packages:

import json
import os
import sys
import pathlib
from glob import glob

import h5py
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import *

from keras.api.applications.vgg16 import VGG16
from keras.api.applications.vgg19 import VGG19
from keras.api.applications.xception import Xception
from keras.api.applications.resnet_v2 import ResNet152V2
from keras.api.applications.inception_resnet_v2 import InceptionResNetV2

from keras.api.preprocessing.image import *
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ch3.recipe1.feature_extractor import FeatureExtractor


# Define the input size of all the feature extractors:
INPUT_SIZE = (224, 224, 3)


# Define a function that will obtain a list of tuples of pre-trained networks, 
# along with the dimensionality of the vectors they output:
def get_pretrained_networks():
    return [
        (VGG16(input_shape=INPUT_SIZE,
               weights='imagenet',
               include_top=False),
         7 * 7 * 512),
        (VGG19(input_shape=INPUT_SIZE,
               weights='imagenet',
               include_top=False),
         7 * 7 * 512),
        (Xception(input_shape=INPUT_SIZE,
                  weights='imagenet',
                  include_top=False),
         7 * 7 * 2048),
        (ResNet152V2(input_shape=INPUT_SIZE,
                     weights='imagenet',
                     include_top=False),
         7 * 7 * 2048),
        (InceptionResNetV2(input_shape=INPUT_SIZE,
                           weights='imagenet',
                           include_top=False),
         5 * 5 * 1536)
    ]


# Define a function that returns a dict of machine learning models to spot-check:
def get_classifiers():
    models = {}
    models['LogisticRegression'] = LogisticRegression()
    models['SGDClf'] = SGDClassifier()
    models['PAClf'] = PassiveAggressiveClassifier()
    models['DecisionTreeClf'] = DecisionTreeClassifier()
    models['ExtraTreeClf'] = ExtraTreeClassifier()

    n_trees = 100
    models[f'AdaBoostClf-{n_trees}'] = \
        AdaBoostClassifier(n_estimators=n_trees)
    models[f'BaggingClf-{n_trees}'] = \
        BaggingClassifier(n_estimators=n_trees)
    models[f'RandomForestClf-{n_trees}'] = \
        RandomForestClassifier(n_estimators=n_trees)
    models[f'ExtraTreesClf-{n_trees}'] = \
        ExtraTreesClassifier(n_estimators=n_trees)
    models[f'GradientBoostingClf-{n_trees}'] = \
        GradientBoostingClassifier(n_estimators=n_trees)

    number_of_neighbors = range(3, 25)
    for n in number_of_neighbors:
        models[f'KNeighborsClf-{n}'] = \
            KNeighborsClassifier(n_neighbors=n)

    reg = [1e-3, 1e-2, 1, 10]
    for r in reg:
        models[f'LinearSVC-{r}'] = LinearSVC(C=r)
        models[f'RidgeClf-{r}'] = RidgeClassifier(alpha=r)

    print(f'Defined {len(models)} models.')
    return models


# Define the path to the dataset, as well as a list of all image paths:
dataset_path = (pathlib.Path.home() / '.keras' / 'datasets' /
                'flowers17')
files_pattern = (dataset_path / 'images' / '*' / '*.jpg')
images_path = [*glob(str(files_pattern))]


# Load the labels into memory:
labels = []
for index in tqdm(range(len(images_path))):
    image_path = images_path[index]
    image = load_img(image_path)

    label = image_path.split(os.path.sep)[-2]
    labels.append(label)

    image.close()


# Define some variables in order to keep track of the spot-checking process. 
# final_report will contain the accuracy of each classifier, trained on the features produced by different pre-trained networks. 
# best_model, best_accuracy, and best_features will contain 
# the name of the best model, its accuracy, and the name of the pre-trained network 
# that produced the features, respectively:
final_report = {}
best_model = None
best_accuracy = -1
best_features = None


# Iterate over each pre-trained network, using it to extract features from the images in the dataset:
for model, feature_size in get_pretrained_networks():
    output_path = dataset_path / f'{model.name}_features.hdf5'
    output_path = str(output_path)
    fe = FeatureExtractor(model=model,
                          input_size=INPUT_SIZE,
                          label_encoder=LabelEncoder(),
                          num_instances=len(images_path),
                          feature_size=feature_size,
                          output_path=output_path)

    fe.extract_features(image_paths=images_path,
                        labels=labels)


    # Take 80% of the data to train, and 20% to test:
    db = h5py.File(output_path, 'r')

    TRAIN_PROPORTION = 0.8
    SPLIT_INDEX = int(len(labels) * TRAIN_PROPORTION)

    X_train, y_train = (db['features'][:SPLIT_INDEX],
                        db['labels'][:SPLIT_INDEX])
    X_test, y_test = (db['features'][SPLIT_INDEX:],
                      db['labels'][SPLIT_INDEX:])

    classifiers_report = {
        'extractor': model.name
    }

    print(f'Spot-checking with features from {model.name}')


    # Using the extracted features in the current iteration, 
    # go over all the machine learning models, 
    # training them on the training set and evaluating them on the test set:
    for clf_name, clf in get_classifiers().items():
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f'\t{clf_name}: {e}')
            continue

        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f'\t{clf_name}: {accuracy}')
        classifiers_report[clf_name] = accuracy


        # Check if we have a new best model. If that's the case, update the proper variables:
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf_name
            best_features = model.name
    

    # Store the results of this iteration in final_report and free the resources of the HDF5 file:
    final_report[output_path] = classifiers_report
    db.close()


# Update final_report with the information of the best model. Finally, write it to disk:
final_report['best_model'] = best_model
final_report['best_accuracy'] = best_accuracy
final_report['best_features'] = best_features

with open('final_report.json', 'w') as f:
    json.dump(final_report, f)
