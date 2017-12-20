#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import csv
from helpers import load_data_and_labels, clean_str

import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

os.environ['KERAS_BACKEND'] = "tensorflow"
# theano'

# TODO better fix of these numbers
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 200

PREDICT = True

positive_data = "./data/preprocess_train_pos_full.txt"
negative_data = "./data/preprocess_train_neg_full.txt"

print("Loading data...")
x_train, labels = load_data_and_labels(positive_data, negative_data)

if PREDICT:
    source_test = open("./data/test_data.txt", "r")

    tests = list()
    for r in source_test:
        r = r.partition(",")[2]
        tests.append(r)

    tests = [s.strip() for s in tests]
    x_test = [clean_str(sent) for sent in tests]
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x_train)
    sequences = tokenizer.texts_to_sequences(x_test)

    word_index = tokenizer.word_index

    print("Load model")
    loaded_model = load_model("./runs/paper2Model.h5")

    print("Calculation of the predictions")
    print("Found", len(word_index), "unique tokens.")

    x_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    z = loaded_model.predict(x_test, verbose=1)
    prediction = np.argmax(z, axis=-1)

    print("rendering of the predictions.csv inside the data/ directory")
    with open("data/predictions.csv", "w") as prediction_file:
        wtr = csv.writer(prediction_file)
        wtr.writerow(("Id", "Prediction"))
        line = 1
        for pred in prediction:
            if pred == 0:
                pred = str(-1)
            else:
                pred = str(1)
            wtr.writerow((line, pred))
            line += 1
