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

    best_perf_of_columns = np.full((4), 0)
    min_poly_degree = 8
    max_poly_degree = 13

    y_clustered, normalTX, ids_clustered, test_y_clustered, normal_test_tx_clustered, test_ids_clustered = prepare_clusters(data_train, data_test)

    for i in range(min_poly_degree, max_poly_degree+1):
        tx_clustered = normalTX
        test_tx_clustered = normal_test_tx_clustered
        coeffArr = np.full((4), i)
        tx_clustered, test_tx_clustered = preprocessing(tx_clustered, test_tx_clustered, coeffArr)
        w, perf_of_columns, predictions = search_best_polynomial_fit(i, y_clustered, tx_clustered, test_y_clustered, test_tx_clustered)
        for index, el in enumerate(perf_of_columns):
            if el > best_perf_of_columns[index]:
                best_perf_of_columns[index] = el
                test_y_clustered[index] = predictions[index]

    # Post Processing
    test_ids = [item for sublist in test_ids_clustered for item in sublist]
    y_pred = [item for sublist in test_y_clustered for item in sublist]

    test_ids, y_pred = zip(*sorted(zip(test_ids, y_pred)))

    # Output Data
    name = 'submission.csv'
    create_csv_submission(test_ids, y_pred, name)
