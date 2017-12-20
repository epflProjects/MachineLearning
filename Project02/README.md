# Machine Learning - Text Classification

This software performs sentiment analysis of tweets and then classify them into 2 groups: positive or negative.

## Requirements
- Python 3
- Keras
- TensorFlow
- Numpy

## Architecture of the Code
The code is contained inside the `cnn/` folder.
- `main.py` : the main script, which will perform predictions on a test file based on a loaded model. See below all the parameters you can set. If you run `python3 main.py`(without parameters) you will obtain our best result.
- `train.py` : file containing a complete training of a convolutional neural network. See below all the parameters you can set.
- `models.py` : file containing methods that implement 2 different convolutional neural networks (CNN).
- `helpers.py` : file containg helping methods for the `main.py` and `train.py`.

## Input Data Requirements
Make sure you have the two files train and test at the right place : './cnn/data/train.txt' and './cnn/data/test.txt'.
The embeddings have to be located: './cnn/embeddings/embeddings.txt'.
The h5 model file have to be located: './cnn/runs/model.h5'.
To obtain our best result you need to have:
- './cnn/data/preprocess_train_pos_full.txt'
- './cnn/data/preprocess_train_neg_full.txt'
- './cnn/embeddings/glove.twitter.27B.200d.txt'
- './cnn/runs/complexModel.h5'

## Output Data Form
The `main.py` outputs a CSV, inside './cnn/data/' directory, containing 2 columns.
- `Id` : the id of the data.
- `Prediction` : `1` if the tweet is evaluated as positive, `-1` as negative.

The `train.py` saves the trained model inside a h5 file at './cnn/runs/'.

## How to run
If you want to create the CSV file with the prediction made on 'cnn/data/test.csv' using a saved model:
`python3 main.py`

Optional parameters:
```
-m_file MODEL_FILE, --model_file MODEL_FILE
                        Name of the h5 file containing the model
  -pos_file POS_FILE, --pos_train_file POS_FILE
                        Name of the positive training file located in the
                        data/ directory
  -neg_file NEG_FILE, --neg_train_file NEG_FILE
                        Name of the negative training file located in the
                        data/ directory
  -test_file TEST_FILE, --test_file TEST_FILE
                        Name of the file that you want to make predictions
```

If you want to train a model:
`python3 train.py`

Optional parameters:
```
-conv CONV_ALGO, --convolution CONV_ALGO
                        Convolution algorithm; you need to choose between
                        'simple_conv' and 'complex_conv'
  -epochs EPOCHS, --numb_epochs EPOCHS
                        Number of epochs
  -pos_file POS_FILE, --pos_train_file POS_FILE
                        Name of the positive training file located in the
                        data/ directory
  -neg_file NEG_FILE, --neg_train_file NEG_FILE
                        Name of the negative training file located in the
                        data/ directory
  -emb_file EMB_FILE, --embeddings_file EMB_FILE
                        name of the embeddings file located inside the
                        embeddings/ folder
  -emb_dim EMB_DIM, --embeddings_dim EMB_DIM
                        Dimension of the embeddings
  -batch_size BATCH_SIZE, --batch_size BATCH_SIZE
                        Size of the batches during the training
```

## Authors
Arnaud Pannatier, Alexander Holloway, Bastian Nanchen

_EPFL Machine Learning CS-433 Course 2017_
