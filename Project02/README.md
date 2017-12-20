# Machine Learning - Text Classification

This software allows to classify data detected in a particles collider into 2 groups depending on the detection or not of the Higgs boson.

## Requirements
- Python 3
- TensorFlow
- Keras
- Numpy

## Architecture of the Code
The code is contained inside the `cnn/` folder.
- `main.py` : the main script
- `train.py` :
- `models.py` :
- `helpers.py` :

## Input Data Requirements
Make sure you have the two files train and test at the right place : './cnn/data/train.csv' and './cnn/data/test.csv'

## Output Data Form
The program outputs a CSV inside `cnn/data` directory containing 2 columns.
- `Id` : the id of the data.
- `Prediction` : `1` if the tweet is evaluated as positive, `-1` as negative.

## How to run
If you want to create the CSV file with the prediction made on `cnn/data/test.csv` using the best trained and saved model:
`python3 main.py`
Optional parameters:
```-m_file MODEL_FILE, --model_file MODEL_FILE
                        Name of the h5 file containing the model
  -pos_file POS_FILE, --pos_train_file POS_FILE
                        Name of the positive training file located in the
                        data/ directory
  -neg_file NEG_FILE, --neg_train_file NEG_FILE
                        Name of the negative training file located in the
                        data/ directory
  -test_file TEST_FILE, --test_file TEST_FILE
                        Name of the file that you want to make predictions```

If you want to train a model:
`python3 train.py`
Optional parameters:
```-conv CONV_ALGO, --convolution CONV_ALGO
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
                        Size of the batches during the training```

## Authors
Arnaud Pannatier, Bastian Nanchen, Alexander Holloway

_EPFL Machine Learning CS-433 Course 2017_
