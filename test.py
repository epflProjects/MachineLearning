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

#test_y_clustered, test_ids_clustered, best_poly_degree_per_group = search_best_polynomial_fit_per_group(data_train, data_test, 9, 12)

best_perf_of_columns = [ 0.84521379, 0.80938564, 0.83928855, 0.83112254]
best_poly_degree_per_group = [ 10, 11, 12, 12] # [9, 11, 11, 12]

test_y_clustered, test_ids_clustered = preprocessing(data_train, data_test, best_perf_of_columns, best_poly_degree_per_group, best_poly_degree_per_group)

print("--- Post Processing ---")
# Post Processing
test_ids = [item for sublist in test_ids_clustered for item in sublist]
y_pred = [item for sublist in test_y_clustered for item in sublist]

test_ids, y_pred = zip(*sorted(zip(test_ids, y_pred)))

# Output Data
name = 'try_submission.csv'
create_csv_submission(test_ids, y_pred, name)
print("--- csv created ---")
