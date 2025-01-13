# TODO
# Import all the necessary packages:
import pathlib

import h5py
from creme import stream
from creme.linear_model import LogisticRegression
from creme.metrics import Accuracy
from creme.multiclass import OneVsRestClassifier
from creme.preprocessing import StandardScaler


# Define a function that will save a dataset as a CSV file:
def write_dataset(output_path, feats, labels, batch_size):
    feature_size = feats.shape[1]
    csv_columns = ['class'] + [f'feature_{i}'
                               for i in range(feature_size)]


    # We'll have one column for the class of each feature, and as many columns of elements in each feature vector. 
    # Next, let's write the contents of the CSV file in batches, starting with the header:
    dataset_size = labels.shape[0]
    with open(output_path, 'w') as f:
        f.write(f'{",".join(csv_columns)}\n')


        # Extract the batch in this iteration:
        for batch_number, index in \
                enumerate(range(0, dataset_size, batch_size)):
            print(f'Processing batch {batch_number + 1} of '
                  f'{int(dataset_size / float(batch_size))}')

            batch_feats = feats[index: index + batch_size]
            batch_labels = labels[index: index + batch_size]
            

            # Now, write all the rows in the batch:
            for label, vector in \
                    zip(batch_labels, batch_feats):
                vector = ','.join([str(v) for v in vector])
                f.write(f'{label},{vector}\n')


# Load the dataset in HDF5 format:
dataset_path = str(pathlib.Path.home() / '.keras' / 'kagglehub' / 'datasets' / 'peterjun' / 'car196' / 'versions' / '2' /
                   'car_ims_rotated' / 'features.hdf5')
db = h5py.File(dataset_path, 'r')


# Define the split index to separate the data into training (80%) and test (20%) chunks:
TRAIN_PROPORTION = 0.8
SPLIT_INDEX = int(db['labels'].shape[0] * TRAIN_PROPORTION)


# Write the training and test subsets to disk as CSV files:
BATCH_SIZE = 256
write_dataset('train.csv',
              db['features'][:SPLIT_INDEX],
              db['labels'][:SPLIT_INDEX],
              BATCH_SIZE)
write_dataset('test.csv',
              db['features'][SPLIT_INDEX:],
              db['labels'][SPLIT_INDEX:],
              BATCH_SIZE)


# creme requires us to specify the type of each column in the CSV file as a dict. instance 
# The following block specifies that class should be encoded as int, 
# while the remaining columns, corresponding to the features, should be of the float type:
FEATURE_SIZE = db['features'].shape[1]
types = {f'feature_{i}': float for i in range(FEATURE_SIZE)}
types['class'] = int


# In the following code, we are defining a creme pipeline, where each input will be standardized prior to being passed to the classifier. 
# Because this is a multi-class problem, we need to wrap LogisticRegression with OneVsRestClassifier:
model = StandardScaler()
model |= OneVsRestClassifier(LogisticRegression())


# Define Accuracy as the target metric and create an iterator over the train.csv dataset:
metric = Accuracy()
dataset = stream.iter_csv('train.csv',
                          target_name='class',
                          converters=types)


# Train the classifier, one example at a time. Print the running accuracy every 100 examples:
print('Training started...')
for i, (X, y) in enumerate(dataset):
    predictions = model.predict_one(X)
    model = model.fit_one(X, y)
    metric = metric.update(y, predictions)

    if i % 100 == 0:
        print(f'Update {i} - {metric}')

print(f'Final - {metric}')


# Create an iterator over the test.csv file:
metric = Accuracy()
test_dataset = stream.iter_csv('test.csv',
                               target_name='class',
                               converters=types)


# Evaluate the model on the test set once more, one sample at a time:
print('Testing model...')
for i, (X, y) in enumerate(test_dataset):
    predictions = model.predict_one(X)
    metric = metric.update(y, predictions)

    if i % 1000 == 0:
        print(f'(TEST) Update {i} - {metric}')

print(f'(TEST) Final - {metric}')
