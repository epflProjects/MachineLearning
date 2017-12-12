import numpy as np

embeddings  = np.load('embeddings.npy')
train_pos   = [line.rstrip() for line in open('twitter-datasets/train_pos.txt')]
train_neg   = [line.rstrip() for line in open('twitter-datasets/train_neg.txt')]
test        = [line.split(',',1)[1].rstrip() for line in open('twitter-datasets/test_data.txt')]
words       = [line.rstrip() for line in open('vocab_cut.txt')]
freq        = [float(line.rstrip()) for line in open('freq.txt')]

train       = train_pos + train_neg
features    = []
labels      = []
testdata    = []

allTweets   = train+test

# TODO: normalise word vectors

# TODO: remove the  
k = -1
print("Starting processing")
for tweet in allTweets:
    k+=1
    if k%1000 == 0:
        print(" ----- ",k, " tweets have been transformed [ ", np.round(k/len(allTweets)*100) ," %]",  end="\r")
    split = str.split(tweet, ' ')
    n_words = len(split)
    sigma = 0
    for word in split:
        try:
            i = words.index(word)

            # keeping the vectors is maybe a better option for learning (before it was np.sum(embeddings)) 
            ## Now taking account of the frequency
            sigma += embeddings[i, ]
        except ValueError:    # the word is not in our vocab
                n_words = n_words - 1
                continue
    try:
        if k < len(train):
            features.append(sigma / n_words)    # get the average of the word vectors
            if k < len(train_pos):
                labels.append(1)
            else :
                labels.append(-1)
        else:
            testdata.append(sigma/n_words)

    except ZeroDivisionError:
        # I handle the case by adding the null vector as the representation of the tweet (maybe we'll change that)
        if k < len(train):
            features.append(embeddings[0, ]*0)    # get the average of the word vectors
            if k < len(train_pos):
                labels.append(1)
            else :
                labels.append(-1)
        else:
            testdata.append(embeddings[0, ]*0)
        continue    # Tweet did not contain any words in our vocab
       
        
print("All tweets have been transformed                                 ")
print("Done ! Saving the files..")

np.save('features', features)
np.save('labels', labels)
np.save('test', testdata)