import pandas as pd
import os

local_path = "/Users/ayush/ml-team-competition"
remote_path = "/home/q-ayushpandey/ml-team-competition/"
os.chdir(local_path)

train = pd.read_csv("train.csv", index_col='Id')
train.head
num_text = train["Text"].size
# Initialize an empty list to hold the clean Text
clean_train_text = []

# Loop over each Text; create an index i that goes from 0 to the length
# of the movie review list 
for i in xrange(0, num_text):
    # Call our function for each one, and add the result to the list of
    # clean Text
    clean_train_text.append(train["Text"][i])
    print i

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=10000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)
# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab

from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit(train_data_features, train["Sentiment"])
