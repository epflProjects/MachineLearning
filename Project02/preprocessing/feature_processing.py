import numpy as np

embeddings = np.load('embeddings.npy')
train_pos  = [line.rstrip() for line in open('twitter-datasets/train_pos.txt')]
train_neg  = [line.rstrip() for line in open('twitter-datasets/train_neg.txt')]
words = [line.rstrip() for line in open('vocab_cut.txt')]

# TODO: normalise word vectors
features = []
for tweet in train_neg:
  split = str.split(tweet, ' ')
  n_words = len(split)
  sigma = 0
  for word in split:
    try:
      i = words.index(word)
      sigma += np.sum(embeddings[i, ])
    except ValueError:  # the word is not in our vocab
        n_words = n_words - 1
        continue
  try:
    features.append(sigma / n_words)  # get the average of the word vectors
  except ZeroDivisionError:
    continue  # Tweet did not contain any words in our vocab
    # TODO: how should we handle these cases?
