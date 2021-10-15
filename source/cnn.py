# This file is part of source.
#
# Celestine is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# source is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# source. If not, see <http://www.gnu.org/licenses/>.
#
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""
This script provides a CNN to solve a classification problem.

The script receives the training and test datasets (labels and data) as well as the mRMR ranking to perform the
classification. The optimisation of the parameters in the different models is carried out through a grid search,
using :py:class:`sklearn.model_selection.GridSearchCV`. Different subsets of features of the mRMR ranking are
selected in order to determine which subset is the best.

"""

import os
import sys

# Python libraries
import numpy as np
from time import time
from sklearn import metrics
import pandas as pd
from pymongo import MongoClient

# Keras and Tensorflow
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
from keras import backend as K
from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2', '3'}


def fit_model(model, data_train, data_test, labels_train, labels_test):
    """
    Fit the model. The features of the dataset (mRMR ranking) are selected before fit the model

    :param model: Classifier model created by the GridSearchCV function
    :type model: :py:class:`sklearn.model_selection.GridSearchCV`

    :param data_train: Training dataset data
    :type data_train: :py:class:`numpy.ndarray`

    :param data_test: Test dataset data
    :type data_test: :py:class:`numpy.ndarray`

    :param labels_train: Training dataset labels
    :type labels_train: :py:class:`numpy.ndarray`

    :param labels_test: Test dataset labels
    :type labels_test: :py:class:`numpy.ndarray`

    :return Accuracy
    :rtype: :py:class:`float`

    """
    model.fit(data_train, labels_train)
    prediction = model.predict(data_test)
    return metrics.accuracy_score(labels_test, prediction)


def create_model(n_filters=15, kernel_size=3):
    """
    Create the CNN model.

    :param n_filters: Number of filters
    :type n_filters: :py:class:`int`

    :param kernel_size: Kernel size
    :type kernel_size: :py:class:`int`

    :return CNN model
    :rtype: :py:class:`keras.model.sequential`

    """
    # Cleaning memory
    K.clear_session()

    # Initialization network weights
    glorot = glorot_uniform(seed=1)

    # Creation of the model
    model = Sequential()

    model.add(Conv1D(n_filters,
                     kernel_size=kernel_size,
                     activation='relu',
                     kernel_initializer=glorot))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    # Learning function optimizer
    adam = Adam(lr=0.001)

    # Multiclass classifier
    model.add(Dense(3, activation='softmax', kernel_initializer=glorot))

    # Compiling the model
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    """
    Main function
    :param None
    :return None
    """

    init_time = time()

    # Prepare database for saving results
    client = MongoClient()
    db = client.Classifiers
    results = db.results

    cnn_parameters = {}

    cnn_parameters['train_data'] = sys.argv[1]
    cnn_parameters['train_labels'] = sys.argv[2]
    cnn_parameters['test_data'] = sys.argv[3]
    cnn_parameters['test_labels'] = sys.argv[4]
    cnn_parameters['mrmr_features'] = sys.argv[5]

    # Read the MRMR file
    mrmr_features = pd.read_csv(cnn_parameters['mrmr_features'], sep=";", header=0)
    features = list(np.array(mrmr_features).reshape(1, 3600)[0])

    # Read the datasets
    data = {
        'train': np.load(cnn_parameters['train_data']), 'test': np.load(cnn_parameters['test_data'])
    }

    labels = {
        'train': to_categorical(np.load(cnn_parameters['train_labels'])),
        'test': np.load(cnn_parameters['test_labels'])
    }

    # Parameters for the grid search and all the necessary data
    features_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    cnn_accuray = {'algorithm': 'cnn'}

    for i in features_range:
        cnn_accuray[str(i)] = []

    param_grid = dict(n_filters=[4, 8, 16, 32, 64, 128], kernel_size=[2, 4, 8, 16, 32], epochs=[25, 50, 75, 100])

    with tf.device('/CPU:0'):
        # Create the model
        model = KerasClassifier(build_fn=create_model)
        grid_cnn = GridSearchCV(estimator=model, param_grid=param_grid)

        for f in features_range:
            data_to_train = np.expand_dims(data['train'][:, features[0:f]], axis=2)
            data_to_test = np.expand_dims(data['test'][:, features[0:f]], axis=2)
            cnn_accuray[str(f)].append(
                fit_model(grid_cnn, data_to_train, data_to_test, labels['train'], labels['test']))

        results.insert_one(cnn_accuray)
