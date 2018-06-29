import pandas as pd
import os
from bs4 import BeautifulSoup
import re

local_path = "/Users/ayush/ml-team-competition"
remote_path = "/home/q-ayushpandey/ml-team-competition/"
os.chdir(local_path)
xl = pd.ExcelFile("train.xlsx")
# xl = pd.ExcelFile("ml-team-competition/train.xlsx")
# xl.sheet_names
train = xl.parse("Consolidated")
train.pop("File")
train.pop("Entity")
train.pop("UMID")
# train_text_sentiment = train[['Text','Sentiment']]

# Combine Multiple Dataframes
# frames = [df1, df2, df3]
# result = pd.concat(frames)

# train_text_sentiment_0 = train_text_sentiment[0:100000]
# train_text_sentiment_0.to_csv("/home/q-ayushpandey/ml-team-competition/train_text_sentiment_0.csv")
# train_text_sentiment_1 = train_text_sentiment[100000:200000]
# train_text_sentiment_1.to_csv("/home/q-ayushpandey/ml-team-competition/train_text_sentiment_1.csv")
# train_text_sentiment_2 = train_text_sentiment[200000:300000]
# train_text_sentiment_2.to_csv("/home/q-ayushpandey/ml-team-competition/train_text_sentiment_2.csv")
# train_text_sentiment_3 = train_text_sentiment[300000:400000]
# train_text_sentiment_3.to_csv("/home/q-ayushpandey/ml-team-competition/train_text_sentiment_3.csv")
# train_text_sentiment_4 = train_text_sentiment[400000:500000]
# train_text_sentiment_4.to_csv("/home/q-ayushpandey/ml-team-competition/train_text_sentiment_4.csv")
# train_text_sentiment_5 = train_text_sentiment[500000:600000]
# train_text_sentiment_5.to_csv("/home/q-ayushpandey/ml-team-competition/train_text_sentiment_5.csv")
# train_text_sentiment_6 = train_text_sentiment[600000:700000]
# train_text_sentiment_6.to_csv("/home/q-ayushpandey/ml-team-competition/train_text_sentiment_6.csv")
# train_text_sentiment_6 = train_text_sentiment[700000:701989]
# train_text_sentiment_7.to_csv("/home/q-ayushpandey/ml-team-competition/train_text_sentiment_7.csv")
# print train["Sentiment"][0]
# print train["Text"][0]
# Remove HTML tag

# example1 = BeautifulSoup(train["Text"][0])  
# print train["Text"][0]
# print example1.get_text()
# Use regular expressions to do a find-and-replace
# Some more cleaning

# Find anything that is NOT a lowercase letter (a-z) or an upper case letter (A-Z), and replace it with a space.
# letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
#                       " ",                   # The pattern to replace it with
#                       example1.get_text() )

# print letters_only
# lower_case = letters_only.lower()        # Convert to lower case
# words = lower_case.split()               # Split into words
import nltk
# nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords  # Import the stop word list

stops = set(stopwords.words("english"))


# "u" before each word; it just indicates that Python is internally representing each word as a unicode string.

# Convert above code in a Function
def text_to_words(raw_text):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_text).get_text()
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    # stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    clear_text = " ".join(meaningful_words)
    train["Text"][i] = clear_text
    print i
    # return( " ".join( meaningful_words ))


# clean_text = text_to_words( train["Text"][0] )
# print clean_text
num_text = train["Text"].size  # 701989
# num_text = 100000
for i in xrange(0, num_text):
    # Call our function for each one, and add the result to the list of
    # clean Text
    text_to_words(train["Text"][i])

# for i in xrange(100000, 200000):
#     # Call our function for each one, and add the result to the list of
#     # clean Text
#     text_to_words(train["Text"][i])

# for i in xrange(200000, 300000):
#     # Call our function for each one, and add the result to the list of
#     # clean Text
#     text_to_words(train["Text"][i])

# for i in xrange(300000, 400000):
#     # Call our function for each one, and add the result to the list of
#     # clean Text
#     text_to_words(train["Text"][i])
# for i in xrange(400000, 500000):
#     # Call our function for each one, and add the result to the list of
#     # clean Text
#     text_to_words(train["Text"][i])

# for i in xrange(500000, 600000):
#     # Call our function for each one, and add the result to the list of
#     # clean Text
#     text_to_words(train["Text"][i])

# for i in xrange(600000, 700000):
#     # Call our function for each one, and add the result to the list of
#     # clean Text
#     text_to_words(train["Text"][i])

# for i in xrange(700000, 701989):
#     # Call our function for each one, and add the result to the list of
#     # clean Text
#     text_to_words(train["Text"][i])


# Get the number of Text based on the dataframe column size
# num_text = train["Text"].size #701989
train.to_csv("full_train.csv")
# train.to_csv("/home/q-ayushpandey/ml-team-competition/train.csv")
# train = pd.csv("/home/q-ayushpandey/ml-team-competition/train.csv")
