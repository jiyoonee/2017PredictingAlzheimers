# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 09:57:02 2017

@author: Owner
"""

import matplotlib.pyplot as plt
import pandas as pd
import statistics as stat

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



# load data
data = pd.read_csv('oasis_longitudinal2.csv')
# Fill missing fields with 0
#data = data.fillna(method='ffill')
# Encode columns into integers 
for c in data.columns:
    data[c] = LabelEncoder().fit_transform(data[c])

accuracy = []
labels = []  
N = 30

columns = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
# columns = ['M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']

classifier = SVC(kernel="linear", C=0.025)
"""
SVC(kernel="linear", C=0.025)

classifier = SVC(C=0.025, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

SVC(C=0.025, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

classifier = LogisticRegression()
classifier = MLPClassifier(hidden_layer_sizes=(100,100), alpha=0.001, 
                           activation="logistic", 
                           max_iter=100000,
                           learning_rate="adaptive"
                           )
classifier = KNeighborsClassifier(2)
classifier = GaussianNB()
classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
classifier = SVC(kernel="linear", C=0.025)


"""

for k in range(N):
    # Split training and test data 7:3
    train, test = train_test_split(data, test_size=0.3)
    train_X = train[columns]
    train_y = train.CDR
    test_X = test[columns]
    test_y = test.CDR
    classifier.fit(train_X, train_y)
    accuracy.append(metrics.accuracy_score(classifier.predict(test_X), test_y))
    labels.append(k)
    #print(k) #track progress

"""
plt.plot(accuracy)
plt.xticks([i for i, e in enumerate(labels)])
"""

print("<<Monte Carlo Cross-Validation>>")
print("mean: ",stat.mean(accuracy))
print("st.dev: ",stat.stdev(accuracy))  
print("max: ",max(accuracy))
print("min: ",min(accuracy)) 

print("<<k-fold Cross-Validation>>")
"""
# determines cv that gives max average score. run once to get cv
maxval=0
maxindex=0
for i in range(2,31):
    scores = cross_val_score(classifier, data[columns], data.CDR, cv=i)
    score = stat.mean(scores)
    if score>maxval:
        maxval = score
        maxindex = i
print(maxindex)
11
"""
scores = cross_val_score(classifier, data[columns], data.CDR, cv=11)
print("mean: ",stat.mean(scores))
print("st.dev: ",stat.stdev(scores))  
print("max: ",max(scores))
print("min: ",min(scores)) 

plt.boxplot([accuracy, scores])
plt.xticks(range(3),('','MonteCarlo','k-fold CV'))