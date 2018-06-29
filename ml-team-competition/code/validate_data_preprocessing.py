import pandas as pd
import os
local_path = "/Users/ayush/ml-team-competition"
remote_path = "/home/q-ayushpandey/ml-team-competition/"
os.chdir(local_path)
xl = pd.ExcelFile("validate.xlsx")
# xl = pd.ExcelFile("ml-team-competition/train.xlsx")
#xl.sheet_names
validate = xl.parse("Human")
validate_text_sentiment = validate[['Message','Human', 'Prediction']]
from bs4 import BeautifulSoup 
import re
import nltk
#nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords # Import the stop word list 
stops = set(stopwords.words("english"))   

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
    validate_text_sentiment["Message"][i] = clear_text 
    print i
    #return( " ".join( meaningful_words ))  

num_text = validate_text_sentiment["Message"].size #701989
# num_text = 100000
for i in xrange(0, num_text):
    # Call our function for each one, and add the result to the list of
    # clean Text
    text_to_words(validate_text_sentiment["Message"][i])

validate_text_sentiment.to_csv("validate.csv")