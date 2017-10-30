from preprocessing import *
from cross_validation import *

def search_best_polynomial_fit_per_group(data_train, data_test, min_degree, max_degree):
    best_perf_of_columns = np.zeros(4)
    best_poly_degree_per_group = np.zeros(4)
    for number_columns in range(min_degree, max_degree + 1):
        degree_per_group = [number_columns, number_columns, number_columns, number_columns]
        (best_perf_of_columns, best_poly_degree_per_group, test_y_clustered), test_ids_clustered = preprocessing(data_train, data_test, best_perf_of_columns, best_poly_degree_per_group, degree_per_group)
    return test_y_clustered, test_ids_clustered, best_poly_degree_per_group

def preprocessing(data_train, data_test, best_per_of_columns, best_number_of_colums, degree_per_group):
    y, tx, ids = data_train
    test_y, test_tx,test_ids = data_test

    PRI_jet_num_colomn_train = tx[:, 22]
    PRI_jet_num_colomn_test = test_tx[:, 22]

    y_clustered = list()
    tx_clustered = list()
    ids_clustered = list()
    test_y_clustered = list()
    test_tx_clustered = list()
    test_ids_clustered = list()

    for i in range(4):
        indices = [ind for ind, a in enumerate(PRI_jet_num_colomn_train) if a == i]

        y_clustered.append(y[indices])
        tx_clustered.append(tx[indices])
        ids_clustered.append(ids[indices])

        test_indices = [ind for ind, a in enumerate(PRI_jet_num_colomn_test) if a == i]

        test_y_clustered.append(test_y[test_indices])
        test_tx_clustered.append(test_tx[test_indices])
        test_ids_clustered.append(test_ids[test_indices])

        # delete colinear columns and preprocess the data
        indices_to_delete = list()
        for col in range(tx_clustered[i].shape[1]):
            if min(tx_clustered[i][:, col]) == max(tx_clustered[i][:, col]):
                indices_to_delete.append(col)

        deleted_tx = np.delete(tx_clustered[i], indices_to_delete, 1)
        tx_clustered[i] = prepro(deleted_tx, int(degree_per_group[i]))#addColumns(deleted_tx, int(best_number_of_colums[i]))

        deleted_tx = np.delete(test_tx_clustered[i], indices_to_delete, 1)
        test_tx_clustered[i] = prepro(deleted_tx, int(degree_per_group[i]))#addColumns(deleted_tx, int(best_number_of_colums[i]))

    return run_cross_validation(tx_clustered, y_clustered, test_tx_clustered, test_y_clustered, best_per_of_columns, best_number_of_colums, degree_per_group[0]), test_ids_clustered



def run_cross_validation(tx_clustered, y_clustered, test_tx_clustered, test_y_clustered, best_per_of_columns, best_number_of_colums, degree):
    w = list()
    result = list()

    loss = 0
    perGood = 0
    maxW = 0

    for i in range(4):
        wi,loss_te, perGoodI = cross_validation_run(tx_clustered[i], y_clustered[i])
        if(perGoodI > best_per_of_columns[i]):
            best_per_of_columns[i] = perGoodI
            best_number_of_colums[i] = degree

        w.append(wi)
        maxW += np.max(np.abs(wi))/4
        loss += loss_te/4
        perGood += perGoodI/4
        test_y_clustered[i] = predict_labels(w[i], test_tx_clustered[i])

    print(" loss : ", loss, "Max of w : ", maxW,  " Percentage of true Y : ", perGood)

    return best_per_of_columns, best_number_of_colums, test_y_clustered
