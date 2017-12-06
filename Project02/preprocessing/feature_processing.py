import numpy as np

embeddings  = np.load('embeddings.npy')
train_pos   = [line.rstrip() for line in open('twitter-datasets/train_pos.txt')]
train_neg   = [line.rstrip() for line in open('twitter-datasets/train_neg.txt')]
words       = [line.rstrip() for line in open('vocab_cut.txt')]

train       = train_pos + train_neg
features    = []
labels      = [];


# TODO: normalise word vectors
k = 0
for tweet in train:

    if k%1000 == 0:
        print(k, " tweets have been transformed")
    split = str.split(tweet, ' ')
    n_words = len(split)
    sigma = 0
    for word in split:
        try:
            i = words.index(word)
            # keeping the vectors is maybe a better option for learning (before it was np.sum(embeddings))
            sigma += embeddings[i, ]
        except ValueError:    # the word is not in our vocab
                n_words = n_words - 1
                continue
    try:
        features.append(sigma / n_words)    # get the average of the word vectors
        if k < len(train_pos):
            labels.append(1)
        else :
            labels.append(0)

    except ZeroDivisionError:
        continue    # Tweet did not contain any words in our vocab
        # TODO: how should we handle these cases?
    k+=1    

np.save('features', features)
np.save('labels', labels)