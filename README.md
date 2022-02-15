# Celestine

Celestine is a Python-based procedure to solve classification problems through five Machine Learning methods:

* K-NN
* SVM
* Naive Bayes
* Random Forest
* Convolutional Neural Network (CNN)

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

Celestine is fully documented in its [github-pages](https://efficomp.github.io/Celestine/). You can also generate its
docs from the source code. Simply change directory to the `doc` subfolder and type in
`make html`, the documentation will be under `build/html`. You will need
[Sphinx](https://www.sphinx-doc.org/en/master/) to build the documentation.

## Usage

There are two files to perform the classification:

* `cnn.py` - Script to run only the CNN.
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

Finally, once the script is finished, the accuracy of each method will be saved in a local database using the PyMongo
library.

## Publications

* J. C. Gómez-López, J. J. Escobar, J. González, F. Gil-Montoya, J. Ortega, M. Burmester, M. Damas. *Energy-Time
  Profiling for Machine Learning Methods to EEG Classification*. In: **International Conference on Bioengineering and
  Biomedical Signal and Image Processing. BIOMESIP 2021**, pp. 311-322. https://doi.org/10.1007/978-3-030-88163-4_27

## Acknowledgments

This work was supported by project *New Computing Paradigms and Heterogeneous Parallel Architectures for High-Performance
and Energy Efficiency of Classification and Optimization Tasks on Biomedical Engineering Applications* 
([HPEE-COBE](https://efficomp.ugr.es/research/projects/hpee-cobe/)), with reference PGC2018-098813-B-C31,
funded by the Spanish *[Ministerio de Ciencia, Innovación y Universidades](https://www.ciencia.gob.es/)*, and by
the [European Regional Development Fund (ERDF)](https://ec.europa.eu/regional_policy/en/funding/erdf/)**.**

<div style="text-align: center">
  <a href="https://www.ciencia.gob.es/">
    <img height="75" src="https://raw.githubusercontent.com/efficomp/ristretto/main/docs/resources/mineco.png" alt="Ministerio de Economía y Competitividad">
  </a> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 
  <a href="https://ec.europa.eu/regional_policy/en/funding/erdf/">
    <img height="75" src="https://raw.githubusercontent.com/efficomp/ristretto/main/docs/resources/erdf.png" alt="European Regional Development Fund (ERDF)">
  </a>
</div>

## License

[GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.md).

## Copyright

Celestine © 2015 [EFFICOMP](https://efficomp.ugr.es).

