from preprocessing import *
from cross_validation import *

def preprocessing(data_train, data_test, best_per_of_columns, best_number_of_colums):
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
        indices = [ind for ind,a in enumerate(PRI_jet_num_colomn_train) if a == i]

        y_clustered.append(y[indices])
        tx_clustered.append(tx[indices])
        ids_clustered.append(ids[indices])

        test_indices = [ind for ind,a in enumerate(PRI_jet_num_colomn_test) if a == i]

        test_y_clustered.append(test_y[test_indices])
        test_tx_clustered.append(test_tx[test_indices])
        test_ids_clustered.append(test_ids[test_indices])

        #delete colinear columns and preprocess the data
        indices_to_delete = list()
        for col in range(tx_clustered[i].shape[1]):
            if min(tx_clustered[i][:, col]) == max(tx_clustered[i][:, col]):
                indices_to_delete.append(col)

        deleted_tx = np.delete(tx_clustered[i], indices_to_delete, 1)
        tx_clustered[i] = prepro(deleted_tx, int(best_number_of_colums[i]))#addColumns(deleted_tx, int(best_number_of_colums[i]))

        deleted_tx = np.delete(test_tx_clustered[i], indices_to_delete, 1)
        test_tx_clustered[i] = prepro(deleted_tx, int(best_number_of_colums[i]))#addColumns(deleted_tx, int(best_number_of_colums[i]))

    return run_cross_validation(tx_clustered, y_clustered, test_tx_clustered, test_y_clustered, best_per_of_columns, best_number_of_colums), test_ids_clustered



def run_cross_validation(tx_clustered, y_clustered, test_tx_clustered, test_y_clustered, best_per_of_columns, best_number_of_colums):

    w = list()
    result = list()

    loss = 0
    perGood = 0
    maxW = 0

    for i in range(4):
        wi,loss_te, perGoodI = cross_validation_run(tx_clustered[i], y_clustered[i])
        if(perGoodI > best_per_of_columns[i]):
            best_per_of_columns[i] = perGoodI
            #best_number_of_colums[i] = nbColumns

        w.append(wi)
        maxW += np.max(np.abs(wi))/4
        loss += loss_te/4
        perGood += perGoodI/4
        test_y_clustered[i] = predict_labels(w[i], test_tx_clustered[i])

    print(" loss : ", loss, "Max of w : ", maxW,  " Percentage of true Y : ", perGood)

    return test_y_clustered












# from preprocessing import *
# from cross_validation import *
#
#
# def search_best_polynomial_fit(poly_degree, y_clustered, tx_clustered, test_y_clustered, test_tx_clustered):
#     """Search the best polynomial fit for each group by cross-validation
#
#     Returns
#         An array of weights, best prediction percentage by group and the predictions"""
#     # Finding the Best columns
#     best_number_of_columns = np.zeros(4)
#     best_per_of_columns = np.zeros(4)
#
#     # Cross-Validation & weights computation
#     w = list()
#     loss = 0
#     perGood = 0
#     maxW = 0
#
#     for i in range(4):
#         wi,loss_te, perGoodI = cross_validation_run(tx_clustered[i], y_clustered[i])
#         if(perGoodI > best_per_of_columns[i]):
#             best_per_of_columns[i] = perGoodI
#             best_number_of_columns[i] = poly_degree
#
#         w.append(wi)
#         maxW += np.max(np.abs(wi))/4
#         loss += loss_te/4
#         perGood += perGoodI/4
#         test_y_clustered[i] = predict_labels(w[i], test_tx_clustered[i])
#
#     print("Number of columns : ", poly_degree, " loss : ", loss, "Max of w : ", maxW,  " Percentage of true Y : ", perGood)
#     print(best_number_of_columns)
#     return w, best_per_of_columns, test_y_clustered
#
#
# def prepare_clusters(data_train, data_test):
#     """Prepare the y, tx and ids clusters for train and test data
#
#     Returns
#         the y, tx and ids clusters for train and test data"""
#
#     y, tx, ids = data_train
#     test_y, test_tx,test_ids = data_test
#
#     PRI_jet_num_colomn_train = tx[:, 22]
#     PRI_jet_num_colomn_test = test_tx[:, 22]
#
#     y_clustered = list()
#     tx_clustered = list()
#     ids_clustered = list()
#     test_y_clustered = list()
#     test_tx_clustered = list()
#     test_ids_clustered = list()
#
#     for i in range(4):
#         indices = [ind for ind,a in enumerate(PRI_jet_num_colomn_train) if a == i]
#
#         y_clustered.append(y[indices])
#         tx_clustered.append(tx[indices])
#         ids_clustered.append(ids[indices])
#
#         test_indices = [ind for ind,a in enumerate(PRI_jet_num_colomn_test) if a == i]
#
#         test_y_clustered.append(test_y[test_indices])
#         test_tx_clustered.append(test_tx[test_indices])
#         test_ids_clustered.append(test_ids[test_indices])
#
#     return y_clustered, tx_clustered, ids_clustered, test_y_clustered, test_tx_clustered, test_ids_clustered
#
#
# def preprocessing(tx_clustered, test_tx_clustered, coeffArr):
#     """Delete colinear columns and preprocess the data
#
#     Returns
#         modified tx_clustered and test_tx_clustered
#         """
#
#     indices_to_delete = list()
#     for i in range(4):
#         for col in range(tx_clustered[i].shape[1]):
#             if min(tx_clustered[i][:, col]) == max(tx_clustered[i][:, col]):
#                 indices_to_delete.append(col)
#
#         deleted_tx = np.delete(tx_clustered[i], indices_to_delete, 1)
#         tx_clustered[i] = prepro(deleted_tx, coeffArr[i])
#
#         deleted_tx = np.delete(test_tx_clustered[i], indices_to_delete, 1)
#         test_tx_clustered[i] = prepro(deleted_tx, coeffArr[i])
#
#     return tx_clustered, test_tx_clustered
