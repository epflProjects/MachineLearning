Generating embeddings
---------------------

1. Clone the `word2vec` and `deepnl` repos

`git clone https://github.com/dav/word2vec`

`git clone https://github.com/attardi/deepnl`

2. Download SemEval 2013 example tweets using `download_tweets.py` (see: [SemEval website](https://www.cs.york.ac.uk/semeval-2013/task2/index.php%3Fid=data.html))
3. Remove quotes around classifications: `sed 's/"//g' sentiment_training.tsv > training.tsv`, where `sentiment_training.tsv` is the file downloaded in the previous step
4. Combine positive and negative Tweets into a sigle file: `cat train_pos.txt train_get.txt > train.txt`
5. Generate embeddings (via `word2vec`, or even using the GloVe script)

`time bin/word2vec -train data/tweets.txt -output data/vectors.txt -cbow 0 -size 200 -window 10 -negative 0 -hs 1 -sample 1e-3 -threads 4 -binary 0`

6. Ensure vocab and vector files match: `cut -d' ' -f1 vectors.txt > vocab.txt`
7. Create the sentiment-specific word embeddings

`python2 bin/dl-sentiwords.py training.tsv --vectors vectors.txt --vocab vocab.txt --variant word2vec -w 10 -s 200 -e 40 --threads 4`

8. Drop the vocab column `cut -d' ' -f2- vectors.txt > sswe.txt`

Notes
-----

* Baseline parameters are selected according to those used in Tang et al. 2014
* Do not be alarmed by the large errors and reported 0.00 accuracy, this is normal and the output embeddings should be useable (see: https://github.com/attardi/deepnl/issues/32)
* The `dl-sentiwords.py` script edits the vector file in-place, so make sure you have a backup
