# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:41:43 2019

@author: rayyan
"""



import numpy as np
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Book1.csv')

dataset=dataset.convert_objects(convert_numeric=True)
dataset=dataset.iloc[:,:].values
#Preprocessing missing data fill up using pearson correlation and then with following process

'''
#coorelation
C=dataset[:,[0,1,5,6,7,8,14,29,30,47,51,52]]
Corr=np.corrcoef(C, rowvar=True)
'''
num_binary = dataset[:,[1,5,6,7,8,9,10,11,12,13,14,15,16,18,21,22,23,24,25,26,30,31,38,39,40,41,42,43,44,45,46,47,49,50,51,52]]
num_binary1 = dataset[0:60,[1,5,6,7,8,9,10,11,12,13,14,15,16,18,21,22,23,24,25,26,30,31,38,39,40,41,42,43,44,45,46,47,49,50,51,52]]
num_binary0 = dataset[60:,[1,5,6,7,8,9,10,11,12,13,14,15,16,18,21,22,23,24,25,26,30,31,38,39,40,41,42,43,44,45,46,47,49,50,51,52]]
num_continuous = dataset[:,[2,3,4,17,19,20,27,28,29,32,33,34,35,36,37,48,53]]
num_continuous1 = dataset[0:60,[2,3,4,17,19,20,27,28,29,32,33,34,35,36,37,48,53]]
num_continuous0 = dataset[60:,[2,3,4,17,19,20,27,28,29,32,33,34,35,36,37,48,53]]
Y= dataset[:,54]
#############################################################

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer1 = imputer.fit(num_binary1[:,:])
num_binary1[:,:] = imputer1.transform(num_binary1[:,:])

imputer0 = imputer.fit(num_binary0[:,:])
num_binary0[:,:] = imputer0.transform(num_binary0[:,:])
#############################################################
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer1 = imputer.fit(num_continuous1[:,:])
num_continuous1[:,:] = imputer1.transform(num_continuous1[:,:])

imputer0 = imputer.fit(num_continuous0[:,:])
num_continuous0[:,:] = imputer0.transform(num_continuous0[:,:])
#############################################################
num_binary[0:60,:]=num_binary1
num_binary[60:,:]=num_binary0

num_continuous[0:60,:]=num_continuous1
num_continuous[60:,:]=num_continuous0

#############################################################
dataset = np.concatenate((num_binary, num_continuous), axis=1)
#############################################################
num_binary = dataset[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]]

num_continuous = dataset[:,[1,14,15,16,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]]
#############################################################
# Statistical Significance Test
#Binary Data
from scipy import stats
n,m=np.shape(num_binary)
Bi_features=np.array([])
alist=[]
for number in range(m):
    A=stats.binom_test(sum(num_binary[:,number]), n=n, alternative='two-sided')
    if A<0.05:
        alist.append(num_binary[:,[number]]) 
Bi_features = np.concatenate(alist, axis=1)  
#features selection test Chi2 test for Continuous data
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import chi2

#feature selection on ANOVA F-value test.......................................
test = SelectFdr(score_func=chi2, alpha=0.05)
fit = test.fit(num_continuous, Y)
'''
np.set_printoptions(precision=3)
print(fit.pvalues_)
'''
con_features = fit.transform(num_continuous)

features = np.concatenate((Bi_features, con_features), axis=1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size = 0.20, random_state = 1)


#Nonlinear Support Vector Machine..............................................
from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', degree=4, random_state = 1, probability=False , tol=1e-3)

classifier.fit(X_train, y_train)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score, LeaveOneOut
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
#print(accuracies)
print("CrossVal Accuracy: %.3f%% " % (accuracies.mean()*100.0))

loocv = LeaveOneOut()
results = cross_val_score(classifier, X=X_train, y=y_train, cv=loocv)
print("LeaveOneOut Accuracy: %.3f%% " % (results.mean()*100.0))


# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
test_acc=accuracy_score(y_test,y_pred)

























