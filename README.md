# Celestine

Celestine is a machine learning Python-based procedure to supervised classification. A total of five 
supervised classifiers have been developed: K-NN, SVM, Naive Bayes, Random Forest, and a Convolutional 
Neural Network (CNN).

This software gives great versatility since its use is not restricted to a specific type of signals, 
but any type classification problem can be solved as long as the datasets have the proper format.

## Requirements

Celestine requires Python 3. It also depends on the following Python packages:

* [NumPy](https://numpy.org/doc/stable/)
* [Pandas](https://pandas.pydata.org/docs/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Keras](https://keras.io/)
* [Tensorflow](https://www.tensorflow.org/)
* [PyMongo](https://pymongo.readthedocs.io/en/stable/)
* [Sphinx](https://www.sphinx-doc.org/en/master/)

## Documentation

Celestine is fully documented in its [github-pages](https://efficomp.github.io/Celestine/). You can also generate
its docs from the source code. Simply change directory to the `doc` subfolder and type in 
`make html`, the documentation will be under `build/html`. You will need 
[Sphinx](https://www.sphinx-doc.org/en/master/) to build the documentation.

## Usage

There are two files to perform the classification:
* `cnn.py` - Script to run the CNN only.
* `classfiers.py` - Script to run the rest of classifiers.

The command to execute each script is as follows:

`$ python3 script data_train.npy labels_train.npy data_test.npy labels_test.npy MRMR.csv` 

where:
* script is `cnn.py` or `classifiers.py`.
* data_train.npy is the file with the training dataset data (in .npy format).
* labels_train.npy is the file with the training dataset labels (in .npy format).
* data_test.npy is the file with the test dataset data (in .npy format).
* labels_test.npy is the file with the test dataset labels (in .npy format).
* MRMR.csv is the file with mRMR features ranking (in .csv format).

Finally, once the scritp is finished, the accuracy will be saved in a local database using the PyMongo library.

## Acknowledgments

This work has been funded by:

* Spanish [*Ministerio de Ciencia, Innovación y Universidades*](https://www.ciencia.gob.es/) under grant number PGC2018-098813-B-C31.
* [*European Regional Development Fund (ERDF)*](https://ec.europa.eu/regional_policy/en/funding/erdf/).

<div style="text-align: right">
  <img src="https://raw.githubusercontent.com/efficomp/Hpmoon/main/docs/logos/mineco.png" height="70">
  <a href="https://www.ciencia.gob.es/">
    <img src="https://raw.githubusercontent.com/efficomp/Hpmoon/main/docs/logos/miciu.jpg" height="70">
  </a>
  <a href="https://ec.europa.eu/regional_policy/en/funding/erdf/">
    <img src="https://raw.githubusercontent.com/efficomp/Hpmoon/main/docs/logos/erdf.png" height="70">
  </a>
</div>

## License

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.md).

## Copyright

Celestine © 2015 [EFFICOMP](https://atcproyectos.ugr.es/efficomp/).

## Publications

1. J. C. Gómez-López, J. J. Escobar, J. González, F. Gil-Montoya, J. Ortega, M. Burmester, and M. Damas.
"Energy-Time Profiling for Machine Learning Methods to EEG Classification". In: *International Conference on 
Bioengineering and Biomedical Signal and Image Processing* (2021), pp. 311-322. doi: 10.1007/978-3-030-88163-4_27