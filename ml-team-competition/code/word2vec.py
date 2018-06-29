# word2vec
# pip install --upgrade gensim
# pip install Cython
# Download the punkt tokenizer for sentence splitting
import pandas as pd
from gensim.models import word2vec
import os
import nltk.data

local_path = "/Users/ayush/ml-team-competition"
remote_path = "/home/q-ayushpandey/ml-team-competition/"
os.chdir(local_path)

train = pd.read_csv("train.csv", index_col='Id')
train.head
num_text = train["Text"].size
print num_text

# nltk.download()


# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def text_to_sentences(text, tokenizer):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(text.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(text)
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for text in train["Text"]:
    sentences += text_to_sentences(text, tokenizer)

print len(sentences)

print sentences[0]

# Set values for various parameters
num_features = 300  # Word vector dimensionality
min_word_count = 1  # Minimum word count
num_workers = 6  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

print "Training model..."

model = word2vec.Word2Vec(sentences, workers=num_workers, \
                          size=num_features, min_count=min_word_count, \
                          window=context, sample=downsampling)
# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using 
# Word2Vec.load()
print "Saving Model...."
model_name = "word2vecmodel"
model.save(model_name)

print "Analyzing Model...."
model.doesnt_match("france england germany berlin".split())

model.most_similar("man")
model.most_similar("awful")
