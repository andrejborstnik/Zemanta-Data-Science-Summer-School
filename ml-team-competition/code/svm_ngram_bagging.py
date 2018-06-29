import pandas as pd
import os
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier

local_path = "/Users/ayush/ml-team-competition"
remote_path = "/home/q-ayushpandey/ml-team-competition/"
os.chdir(local_path)

train = pd.read_csv("train.csv",index_col='Id')
train_1 = train[0:10000]
#train.head
num_text = train["Text"].size
#print num_text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#vectorizer = TfidfVectorizer(min_df=2,max_df = 0.95,sublinear_tf=True,use_idf=True)
vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
train_vectors = vectorizer.fit_transform(train["Text"].values.astype('U'))
train_vectors_1 = vectorizer.fit_transform(train_1["Text"].values.astype('U'))
#print train_vectors_1.size
print train_vectors_1.size
validate = pd.read_csv("validate.csv",index_col='Id')
validate_num_text = validate["Message"].size
test_vectors = vectorizer.transform(validate["Message"].values.astype('U'))

# Model

# svm_rbf = svm.SVC(C = 1, gamma = 0.001, kernel='linear')
n_estimators = 3
svm_rbf = OneVsRestClassifier(BaggingClassifier(svm.SVC(C = 1, gamma = 0.001, kernel='linear'),max_samples= 1.0/n_estimators, n_estimators=n_estimators))
model = svm_rbf.fit(train_vectors,train["Sentiment"])

# param_grid = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, make_scorer
# f1_scorer = make_scorer(f1_score,average=None)
# precision_scorer = make_scorer(precision_score,average=None)
# recall_scorer = make_scorer(recall_score,average=None)
# scoring = [f1_scorer, recall_scorer,precision_scorer]
# grid_clf = GridSearchCV(svm_rbf, param_grid, cv=5, n_jobs=-1)
# #scores = cross_validate(clf, iris.data, iris.target, scoring=scoring, cv=5, return_train_score=False)
# grid_clf.fit(train_vectors, train["Sentiment"])


# print "Cross-Validation done"
# best_model = grid_clf.best_estimator_
# print grid_clf.best_params_
# print grid_clf.grid_scores_

result = model.predict(test_vectors)


actual_prediction = pd.DataFrame( data={"Prediction":result, "Actual":validate["Human"]} )
actual_prediction.to_csv("svm_ngram_bagging.csv")
print "Result written to DataFrame"