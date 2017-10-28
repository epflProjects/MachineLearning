from preprocessing import *
from cross_validation import *
from run import *

"""Data Loading"""
oy, otx, oids = load_csv_data('Data/train.csv', False)
otest_y, otest_tx, otest_ids= load_csv_data('Data/test.csv', False)

#TODO create a method!

# Finding the Best columns
best_number_of_colums = np.zeros(4)
best_per_of_columns = np.zeros(4)

for nbColumns in range(2,15):
#for nbColumns in [2, 5, 10]:
    y, tx, ids = oy, otx, oids
    test_y, test_tx,test_ids = otest_y, otest_tx, otest_ids
    # Preprocessing

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
        tx_clustered[i] = prepro(deleted_tx, nbColumns)

        deleted_tx = np.delete(test_tx_clustered[i], indices_to_delete, 1)
        test_tx_clustered[i] = prepro(deleted_tx, nbColumns)


    # Cross-Validation & weights computation
    w = list()
    result = list()

    loss = 0
    perGood = 0
    maxW = 0
    for i in range(4):
        wi,loss_te, perGoodI = cross_validation_run(tx_clustered[i], y_clustered[i])
        if(perGoodI > best_per_of_columns[i]):
            best_per_of_columns[i] = perGoodI
            best_number_of_colums[i] = nbColumns

        w.append(wi)
        maxW += np.max(np.abs(wi))/4
        loss += loss_te/4
        perGood += perGoodI/4
        test_y_clustered[i] = predict_labels(w[i], test_tx_clustered[i])

    print("Number of columns : ", nbColumns, " loss : ", loss, "Max of w : ", maxW,  " Percentage of true Y : ", perGood)
print(best_number_of_colums)
