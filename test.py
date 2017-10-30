from proj1_helpers import *
from implementations import *
from cross_validation import *
from preprocessing import *
from experimentation import *
import numpy as np

import os
pathTrain = "./Data/train.csv"
pathTest = "./Data/test.csv"

if not (os.path.isfile(pathTrain) and os.path.isfile(pathTest)):
    print("Error : The data files were not found")
    print("Expected : ./Data/train.csv and ./Data/test.csv")
    print("At least one file is missing")
    print("Make sure you have added them in ./Data/")
else:

    #Data Loading
    data_train = []
    data_test = []
    data_train = load_csv_data('Data/train.csv', False)
    data_test = load_csv_data('Data/test.csv', False)
    print("--- Data Loaded ---")

    """Uncomment this line to search the best polynomial degree per PRI_jet_num groups using cross-validation and ridge regression
        This function takes about 60 minutes with a good laptop"""
    #test_y_clustered, test_ids_clustered, best_poly_degree_per_group = search_best_polynomial_fit_per_group(data_train, data_test, 9, 12)

    null_array = [0, 0, 0, 0]
    """Results of the search; search_best_polynomial_fit_per_group function"""
    best_poly_degree_per_group = [ 10, 11, 12, 12]

    """Compute the predictions in searching for the optimal lambda for ridge regression
        This function takes about 17 minutes with a good laptop"""
    test_y_clustered, test_ids_clustered = preprocessing(data_train, data_test, null_array, null_array, best_poly_degree_per_group)

    print("--- Post Processing ---")
    # Post Processing
    test_ids = [item for sublist in test_ids_clustered for item in sublist]
    y_pred = [item for sublist in test_y_clustered for item in sublist]

    test_ids, y_pred = zip(*sorted(zip(test_ids, y_pred)))

    # Output Data
    name = 'submission.csv'
    create_csv_submission(test_ids, y_pred, name)
    print("--- csv created ---")
