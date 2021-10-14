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
This script provides the four classifiers K-NN, SVM, Random Forest, and Naive Bayes to solve a classification problem.

The script receives the training and test datasets (labels and data) as well as the mRMR ranking to perform the
classification. The optimisation of the parameters in the different models is carried out through a grid search,
using :py:class:`sklearn.model_selection.GridSearchCV`. Different subsets of features of the mRMR ranking are
selected in order to determine which subset is the best.

"""

import sys

# Python libraries
import pandas as pd
import numpy as np
from pymongo import MongoClient

# Sklearn
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

__author__ = 'Juan Carlos Gómez-López'
__copyright__ = 'Copyright 2021, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.1.0'
__maintainer__ = 'Juan Carlos Gómez-López'
__email__ = 'goloj@ugr.es'
__status__ = 'Development'


def fit_model(model, data_train, data_test, labels_train, labels_test, features_mrmr, max_features):
    """
    Fit the model.

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

    :param features_mrmr: mRMR ranking
    :type features_mrmr: :py:class:`list`

    :param max_features: First selected features of the mRMR ranking
    :type max_features: :py:class:`int`

    :return Accuracy
    :rtype: :py:class:`float`

    """
    model.fit(data_train[:, features_mrmr[0:max_features]], np.ravel(labels_train))
    prediction = model.predict(data_test[:, features_mrmr[0:max_features]])
    return metrics.accuracy_score(labels_test, prediction)


if __name__ == "__main__":
    # Prepare database for saving results
    client = MongoClient()
    db = client.Classifiers
    db.results.drop()
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
        'train': np.load(cnn_parameters['train_labels']), 'test': np.load(cnn_parameters['test_labels'])
    }

    # Parameters for the grid search and all the necessary data
    parameters_svm = {'C': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5],
                      'gamma': [0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.25, 0.5, 0.75,
                                0.9]}

    parameters_knn = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]}

    parameters_rf = {'n_estimators': [16, 32, 64, 128, 256]}

    features_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 100]

    knn_accuray = {'algorithm': 'knn'}
    svm_accuray = {'algorithm': 'svm'}
    rf_accuray = {'algorithm': 'rf'}
    nb_accuray = {'algorithm': 'nb'}

    for i in features_range:
        knn_accuray[str(i)] = []
        svm_accuray[str(i)] = []
        rf_accuray[str(i)] = []
        nb_accuray[str(i)] = []

    # KNN
    knn = KNeighborsClassifier()
    gs_knn = GridSearchCV(knn, parameters_knn)

    # SVM
    svm = svm.SVC()
    gs_svm = GridSearchCV(svm, parameters_svm)

    # Random Forest
    rf = RandomForestClassifier()
    gs_rf = GridSearchCV(rf, parameters_rf)

    # Naive Bayes
    nb = GaussianNB()

    for f in features_range:
        knn_accuray[str(f)].append(fit_model(model=gs_knn, data_train=data['train'], data_test=data['test'],
                                             labels_train=labels['train'], labels_test=labels['test'],
                                             features_mrmr=features, max_features=f))

        svm_accuray[str(f)].append(fit_model(model=gs_svm, data_train=data['train'], data_test=data['test'],
                                             labels_train=labels['train'], labels_test=labels['test'],
                                             features_mrmr=features, max_features=f))

        rf_accuray[str(f)].append(fit_model(model=gs_rf, data_train=data['train'], data_test=data['test'],
                                            labels_train=labels['train'], labels_test=labels['test'],
                                            features_mrmr=features, max_features=f))

        nb_accuray[str(f)].append(fit_model(model=nb, data_train=data['train'], data_test=data['test'],
                                            labels_train=labels['train'], labels_test=labels['test'],
                                            features_mrmr=features, max_features=f))
    results.insert_one(knn_accuray)
    results.insert_one(svm_accuray)
    results.insert_one(rf_accuray)
    results.insert_one(nb_accuray)
