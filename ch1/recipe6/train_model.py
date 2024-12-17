# Import the necessary packages:
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# pip install tensorflow-docs
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.api.datasets import fashion_mnist as fm

from keras.src.layers import BatchNormalization
from keras.src.layers import Conv2D
from keras.src.layers import Dense
from keras.src.layers import Dropout
from keras.src.layers import ELU
from keras.src.layers import Flatten
from keras.src.layers import Input
from keras.src.layers import MaxPooling2D
from keras.src.layers import Softmax

from keras.src.saving import load_model
from keras.src.models import Model
from keras.src.utils import plot_model


# Define a function that will load and prepare the dataset. 
# It will normalize the data, one-hot encode the labels, 
# take a portion of the training set for validation, 
# and wrap the three data subsets into three separate tf.data.Dataset instances 
# to increase performance using from_tensor_slices():
def load_dataset():
    (X_train, y_train), (X_test, y_test) = fm.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape grayscale to include channel dimension.
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    (X_train, X_val,
     y_train, y_val) = train_test_split(X_train, y_train,
                                        train_size=0.8)

    train_ds = (tf.data.Dataset
                .from_tensor_slices((X_train, y_train)))
    val_ds = (tf.data.Dataset
              .from_tensor_slices((X_val, y_val)))
    test_ds = (tf.data.Dataset
               .from_tensor_slices((X_test, y_test)))

    return train_ds, val_ds, test_ds


# Implement a function that will build a network similar to LeNet 
# with the addition of BatchNormalization, which we'll use to make the network faster and most stable, 
# and Dropout layers, which will help us combat overfitting, 
# a situation where the network loses generalization power due to high variance:
def build_network():
    input_layer = Input(shape=(28, 28, 1))
    x = Conv2D(filters=20,
               kernel_size=(5, 5),
               padding='same',
               strides=(1, 1))(input_layer)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2),
                     strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Conv2D(filters=50,
               kernel_size=(5, 5),
               padding='same',
               strides=(1, 1))(x)
    x = ELU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2),
                     strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(units=500)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)

    x = Dense(10)(x)
    output = Softmax()(x)

    model = Model(inputs=input_layer, outputs=output)
    return model


# Define a function that takes a model's training history, along with a metric of interest, 
# to create a plot corresponding to the training and validation of the curves of such a metric:
def plot_model_history(model_history, metric, ylim=None):
    plt.style.use('seaborn-darkgrid')
    plotter = tfdocs.plots.HistoryPlotter()
    plotter.plot({'Model': model_history}, metric=metric)

    plt.title(f'{metric.upper()}')
    if ylim is None:
        plt.ylim([0, 1])
    else:
        plt.ylim(ylim)

    plt.savefig(f'{metric}.png')
    plt.close()


# Consume the training and validation datasets in batches of 256 images at a time. 
# The prefetch() method spawns a background thread that populates a buffer of size 1024 with image batches:
BATCH_SIZE = 256
BUFFER_SIZE = 1024

train_dataset, val_dataset, test_dataset = load_dataset()

train_dataset = (train_dataset
                 .shuffle(buffer_size=BUFFER_SIZE)
                 .batch(BATCH_SIZE)
                 .prefetch(buffer_size=BUFFER_SIZE))
val_dataset = (val_dataset
               .batch(BATCH_SIZE)
               .prefetch(buffer_size=BUFFER_SIZE))
test_dataset = test_dataset.batch(BATCH_SIZE)


# Build and train the network:
EPOCHS = 100

model = build_network()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model_history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)


# Plot the training and validation loss and accuracy:
plot_model_history(model_history, 'loss', [0., 2.0])
plot_model_history(model_history, 'accuracy')
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')


# Save the model:
model.save('model.hdf5')


# Load and evaluate the model:
loaded_model = load_model('model.hdf5')
results = loaded_model.evaluate(test_dataset)
print(f'Loss: {results[0]}, Accuracy: {results[1]}')
