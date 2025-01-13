# Import the necessary packages:
import pathlib

import h5py
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report


# Load the dataset in HDF5 format:
dataset_path = str(pathlib.Path.home() / '.cache' / 'kagglehub' / 'datasets' /
                   'peterjun' / 'car196' / 'versions' / '2' /
                   'car_ims_rotated' / 'features.hdf5')
db = h5py.File(dataset_path, 'r')


# Because the dataset is too big, we'll only work with 50% of the data.
# The following block splits both the features and labels in half:
SUBSET_INDEX = int(db['labels'].shape[0] * 0.5)
features = db['features'][:SUBSET_INDEX]
labels = db['labels'][:SUBSET_INDEX]
label_names = [label.decode('utf-8') for label in db['label_names']]


# Take the first 80% of the data to train the model, and the remaining 20% to evaluate it later on:
TRAIN_PROPORTION = 0.8
SPLIT_INDEX = int(len(labels) * TRAIN_PROPORTION)
X_train, y_train = (features[:SPLIT_INDEX],
                    labels[:SPLIT_INDEX])
X_test, y_test = (features[SPLIT_INDEX:],
                  labels[SPLIT_INDEX:])


# Train a cross-validated Logistic Regression model on the training set. 
# LogisticRegressionCV will find the best C parameter using cross-validation:
# Notice that n_jobs=-1 means we'll use all available cores to find the best model in parallel. 
# You can adjust this value based on the capacity of your hardware.
model = LogisticRegressionCV(n_jobs=-1)
model.fit(X_train, y_train)


# Evaluate the model on the test set. 
# We'll compute a classification report to get a fine-grained view of the model's performance:
predictions = model.predict(X_test)
report = classification_report(y_test, predictions,
                               target_names=label_names)
print(report)


# Finally, close the HDF5 file to free up any resources:
db.close()
