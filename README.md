# K Nearest Neighbour Imputation on ICU Dataset

Example implementation of K Nearest Neighbour Imputation on ICU dataset, developed as part of a project for University of Bristol COMSM0017 Applied Data Science, attempting to produce a machine learning mdoel capable of predicting sepis 6 hours before diagnosis, in accordance with the [PhysioNet Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/). 

The project report is visible at {?}

* Author: Angus Redlarski Williams
* Github: @angusrw
* Email: angusrwilliams@gmail.com

---

The dataset provided in the [PhysioNet Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) contains a lot of null measurements, with many columns missing over 95% of data. In order to train a model to accurately predict sepsis, this missing data can be imputed. `main.py` implements a K Nearest Neighbours imputation, drawing on [sklearn's imputation library](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) and using a sampling approach (where only a subset of entries are used as possible neighbours) to dramatically reduce the time taken to complete imputation (it should take ~2hrs).

The following packages are required:
* pandas
* numpy
* sklearn

To run on dataset A, execute the command `python3 main.py a`, and likewise for dataset B.

