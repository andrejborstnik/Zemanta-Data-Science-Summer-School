import pandas as pd
import os

local_path = "/Users/ayush/ml-team-competition"
remote_path = "/home/q-ayushpandey/ml-team-competition/"
os.chdir(local_path)

actual_prediction = pd.read_csv("svm_ngram_10.csv",index_col='Id')
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
calculated_f1_score = f1_score(actual_prediction["Actual"], actual_prediction["Prediction"], average=None)
print calculated_f1_score
calculated_precision_score = precision_score(actual_prediction["Actual"], actual_prediction["Prediction"], average=None)
print calculated_precision_score
calculated_recall_score = recall_score(actual_prediction["Actual"], actual_prediction["Prediction"], average=None)
print calculated_recall_score

file = open("svm_ngram_10_score.txt","w") 
 
file.write("f1_score\n")
file.write("%s\n" % calculated_f1_score)
file.write("precision_score\n")
file.write("%s\n" % calculated_precision_score)
file.write("recall_score\n")
file.write("%s\n" % calculated_recall_score)

target_names = ['Nagative', 'Neutral', 'Positive']
classification_report = classification_report(actual_prediction["Actual"], actual_prediction["Prediction"], target_names=target_names)
print classification_report
file.write("%s\n" % classification_report)

file.close()

#svm_rbf = BaggingClassifier(svm.SVC(C = 1, gamma = 0.001, kernel='linear'),max_samples=1.0 / n_estimators, n_estimators=n_estimators)