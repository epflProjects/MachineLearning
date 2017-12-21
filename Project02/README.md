# Machine Learning - Text Classification

This software performs sentiment analysis of tweets and then classify them into 2 groups: positive or negative.

## TL;TR
In order to obtain the predictions of our best model on `test_data.txt`, you need previously to be sure to have on your computer wget. If it is not the case: `brew install wget`.
Then you can run `./run.sh` inside the `src/` directory. The `predictions.csv` is created inside the `/src/data/` directory.

## Requirements
# Dependencies
- Python 3
- Keras
- TensorFlow
- Numpy

# Files
- Stanford Pretrained Glove Word Embeddings :
- Training Data :
- Test Data :

## Architecture of the Code
The code is contained inside the `src/` folder.
- `run.py` : the main script, which will perform predictions on a test file based on a loaded model. The script should take a bit less than 10 minutes to complete. See below all the parameters you can set. If you run `python3 run.py`(without parameters) you will obtain our best result.
- `train.py` : file containing a complete training of a convolutional neural network. See below all the parameters you can set.
- `models.py` : file containing methods that implement 2 different convolutional neural networks (CNN).
- `helpers.py` : file containing helping methods for the `run.py` and `train.py`.

The `preprocessing` folder contains all the files used to generate the embeddings. They are not used in the four python files above.

The `experimentation` folder contains all files, which are not used to obtain our best result, but were useful during our seek to obtain a good CNN.

## Embeddings
The embeddings reside in the `src/embeddings/` directory.
This project used two embeddings:
- `sentiment.txt` : it is the sentiment word embeddings file.
- `glove.twitter.27B.200d.txt` : Stanford Pretrained Glove Word Embeddings by https://nlp.stanford.edu/projects/glove/

## Input Data Requirements
Make sure you have the two files train and test at the right place : `./src/data/`.
The embeddings have to be located: `./src/embeddings/`.
The h5 model file have to be located: `./src/runs/`.
To obtain our best result you need to have:
- `./src/data/preprocess_train_pos_full.txt`
- `./src/data/preprocess_train_neg_full.txt`
- `./src/embeddings/glove.twitter.27B.200d.txt`
- `./src/runs/cnnModel.h5`

## Output Data Form
The `run.py` outputs a CSV, inside `./src/data/` directory, containing 2 columns.
- `Id` : the id of the data.
- `Prediction` : `1` if the tweet is evaluated as positive, `-1` as negative.

The `train.py` saves the trained model inside a h5 file at './src/runs/'.

## How to run
If you want to create the CSV file with the prediction made on `./src/data/test.csv` using a saved model:
`python3 run.py`

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
By default: `python3 run.py -m_file paper2Model.h5 -pos_file preprocess_train_pos_full.txt -neg_file preprocess_train_neg_full -test_file test_data.txt`

If you want to train a model:
`python3 train.py`

Optional parameters:
```
-cnn CONV_ALGO, --cnn_algo CONV_ALGO
                        Convolution algorithm; you need to choose between
                        'simple_conv', 'cnn_without_dropout' and 'cnn_dropout'
                        (by default)
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
By default: `python3 train.py -cnn cnn_dropout -epochs 10 -pos_file preprocess_train_pos_full.txt -neg_file preprocess_train_neg_full -emb_file glove.twitter.27B.200d.txt -emb_dim 200 -batch_size 64`

## Authors
Arnaud Pannatier, Alexander Holloway, Bastian Nanchen

_EPFL Machine Learning CS-433 Course 2017_
