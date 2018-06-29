import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

local_path = "/Users/ayush/ml-team-competition"
remote_path = "/home/q-ayushpandey/ml-team-competition/"
os.chdir(local_path)

train = pd.read_csv("train.csv",index_col='Id')
train.head
num_text = train["Text"].size
print num_text
# Initialize an empty list to hold the clean Text
# clean_train_text = []

# # Loop over each Text; create an index i that goes from 0 to the length
# # of the movie review list 
# for i in xrange(0, num_text):
#     # Call our function for each one, and add the result to the list of
#     # clean Text
#     if train["Text"][i]==None:
#         train["Text"][i]=""
#         print "None String"
    
#     clean_train_text.append(train["Text"][i])
#     print i


vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 1000, ngram_range=(2, 2)) 


# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(train["Text"].values.astype('U'))
# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()
# Take a look at the words in the vocabulary
# vocab = vectorizer.get_feature_names()
# print vocab
print "train_data_features prepared"
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()

# 10-fold Cross-Validation using Grid Search 
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, make_scorer
f1_scorer = make_scorer(f1_score,average=None)
precision_scorer = make_scorer(precision_score,average=None)
recall_scorer = make_scorer(recall_score,average=None)
scoring = [f1_scorer, recall_scorer, precision_scorer]
param_grid = {'n_estimators': [100,150,200,250]}
from sklearn.grid_search import GridSearchCV
grid_clf = GridSearchCV(forest, param_grid, cv=3, n_jobs=-1)
#scores = cross_validate(clf, iris.data, iris.target, scoring=scoring, cv=5, return_train_score=False)
grid_clf.fit(train_data_features, train["Sentiment"])

print "Cross-Validation done"
best_model = grid_clf.best_estimator_
print grid_clf.best_params_
print grid_clf.grid_scores_

# Initialize a Random Forest classifier with 100 trees
#forest = RandomForestClassifier(n_estimators = 50) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
#forest = forest.fit(train_data_features,train["Sentiment"] )

# Predict using the existing model
validate = pd.read_csv("validate.csv",index_col='Id')
clean_validate_text = []
validate_num_text = validate["Message"].size

# Loop over each Text; create an index i that goes from 0 to the length
# of the movie review list 
# for i in xrange(0, validate_num_text):
#     # Call our function for each one, and add the result to the list of
#     # clean Text
#     clean_validate_text.append(validate["Message"][i])
#     print i

#from sklearn.feature_extraction.text import CountVectorizer
test_data_features = vectorizer.transform(validate["Message"].values.astype('U'))
test_data_features = test_data_features.toarray()
result = best_model.predict(test_data_features)
actual_prediction = pd.DataFrame( data={"Prediction":result, "Actual":validate["Human"]} )
actual_prediction.to_csv("rf.csv")
print "Result written to DataFrame"

# from sklearn.metrics import f1_score, precision_score, recall_score
# f1_score(actual_prediction["Actual"], actual_prediction["Prediction"], average=None)
# precision_score(actual_prediction["Actual"], actual_prediction["Prediction"], average=None)
# recall_score(actual_prediction["Actual"], actual_prediction["Prediction"], average=None)


