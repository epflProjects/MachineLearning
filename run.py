from proj1_helpers import *
from implementations import *
from cross_validation import *
from preprocessing import *
from experimentation import *
import numpy as np

#Data Loading
data_train = []
data_test = []
data_train = load_csv_data('Data/train.csv', False)
data_test = load_csv_data('Data/test.csv', False)
print("--- Data Loaded ---")

best_perf_of_columns = [0, 0, 0, 0] #np.full((4), 0)
min_poly_degree = 10
max_poly_degree = 12


y_clustered, initial_tx_clustered, ids_clustered, test_y_clustered, initial_test_tx_clustered, test_ids_clustered = prepare_clusters(data_train, data_test)

for i in range(min_poly_degree, max_poly_degree+1):
    tx_clustered = initial_tx_clustered
    print("before: ",len(initial_tx_clustered))
    test_tx_clustered = initial_test_tx_clustered
    coeffArr = [i, i, i, i] #np.full((4), i)
    tx_clustered, test_tx_clustered = preprocessing(tx_clustered, test_tx_clustered, coeffArr)
    print("after:", len(initial_tx_clustered), len(tx_clustered))
    w, perf_of_columns, predictions = search_best_polynomial_fit(i, y_clustered, tx_clustered, test_y_clustered, test_tx_clustered)
    for index, el in enumerate(perf_of_columns):
        # TODO print(el, ">", best_perf_of_columns[index])
        if el > best_perf_of_columns[index]:
            best_perf_of_columns[index] = el
            test_y_clustered[index] = predictions[index]

print(best_perf_of_columns)
# Post Processing
test_ids = [item for sublist in test_ids_clustered for item in sublist]
y_pred = [item for sublist in test_y_clustered for item in sublist]
print("--- Post Processing ---")

test_ids, y_pred = zip(*sorted(zip(test_ids, y_pred)))
print("--- csv created ---")

# Output Data
name = 'test_perf_submission.csv'
create_csv_submission(test_ids, y_pred, name)
