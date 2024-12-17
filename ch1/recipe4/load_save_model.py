# Import the necessary packages:
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.api.datasets import mnist
from keras.src.models import Model

from keras.src.layers import BatchNormalization
from keras.src.layers import Conv2D
from keras.src.layers import Dense
from keras.src.layers import Dropout
from keras.src.layers import Flatten
from keras.src.layers import Input
from keras.src.layers import MaxPooling2D
from keras.src.layers import ReLU
from keras.src.layers import Softmax

from keras.src.saving import load_model
from keras.src.models.model import model_from_json

# Define a function that will download and prepare the data by normalizing the train and test sets and one-hot encoding the labels:
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize data.
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape grayscale to include channel dimension.
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    # Process labels.
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    return X_train, y_train, X_test, y_test


# Define a function for building a network. 
# The architecture comprises a single convolutional layer and two fully connected layers:
def build_network():
    input_layer = Input(shape=(28, 28, 1), name='input_layer')
    convolution_1 = Conv2D(kernel_size=(2, 2),
                           padding='same',
                           strides=(2, 2),
                           filters=32,
                           name='convolution_1')(input_layer)
    activation_1 = ReLU(name='activation_1')(convolution_1)
    batch_normalization_1 = BatchNormalization(name='batch_normalization_1')(activation_1)
    pooling_1 = MaxPooling2D(pool_size=(2, 2),
                             strides=(1, 1),
                             padding='same',
                             name='pooling_1')(batch_normalization_1)
    dropout = Dropout(rate=0.5, name='dropout')(pooling_1)

    flatten = Flatten(name='flatten')(dropout)
    dense_1 = Dense(units=128, name='dense_1')(flatten)
    activation_2 = ReLU(name='activation_2')(dense_1)
    dense_2 = Dense(units=10, name='dense_2')(activation_2)
    output = Softmax(name='output')(dense_2)

    network = Model(inputs=input_layer, outputs=output, name='my_model')

    return network


# Implement a function that will evaluate a network using the test set:
def evaluate(model, X_test, y_test):
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Accuracy: {accuracy}')


# Prepare the data, create a validation split, and instantiate the neural network:
print('Loading and pre-processing data.')
X_train, y_train, X_test, y_test = load_data()

# Split dataset.
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)

# Build network.
model = build_network()


# Compile and train the model for 50 epochs, with a batch size of 1024. 
# Feel free to tune these values according to the capacity of your machine:
print('Training network...')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=40, batch_size=1024)


# Save the model, along with its weights, in HDF5 format using the save() method. 
# Then, load the persisted model using load_model() and evaluate the network's performance on the test set:
print('Saving model and weights as HDF5.')
model.save('model_and_weights.hdf5')

print('Loading model and weights as HDF5.')
loaded_model = load_model('model_and_weights.hdf5')

print('Evaluating using loaded model.')
evaluate(loaded_model, X_test, y_test)
