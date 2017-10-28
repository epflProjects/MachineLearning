from proj1_helpers import *
from implementations import *
from cross_validation import *
from preprocessing import *
import numpy as np

"""Data Loading"""
oy, otx, oids = load_csv_data('Data/train.csv', False)
otest_y, otest_tx, otest_ids= load_csv_data('Data/test.csv', False)

# TODO put in preprocessing.py?
def prepro(M,c):
    return addFunckyThings(addColumns(M, c), M)
