# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:40:51 2019

@author: rayyan
"""

import numpy as np
import pandas as pd
from scipy import stats


# Importing the dataset
dataset = pd.read_csv('Book1_final_algo3final.csv')

dataset=dataset.convert_objects(convert_numeric=True)
dataset=dataset.iloc[:,:].values
#Preprocessing missing data fill up using pearson correlation and then with following process

'''
#coorelation
C=dataset[:,[0,1,5,6,7,8,14,29,30,47,51,52]]
Corr=np.corrcoef(C, rowvar=True)
'''
num_binary = dataset[:,[1,5,6,7,8,9,10,11,12,13,14,15,16,18,21,22,23,24,25,26,36,37,38,39,40,41,42,43,44,45,47,48,49]]
num_continuous = dataset[:,[2,3,4,17,19,20,27,28,29,30,31,32,33,34,35,46]]
#Y= dataset[:,52]
#############################################################

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer1 = imputer.fit(num_binary[:,:])
num_binary[:,:] = imputer1.transform(num_binary[:,:])

#############################################################
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer1 = imputer.fit(num_continuous[:,:])
num_continuous[:,:] = imputer1.transform(num_continuous[:,:])



#############################################################
se = np.concatenate((num_binary, num_continuous), axis=1)
dataset[:,1:50]=se
################## Train/Test dataset #######################
#data1=dataset[[0,1,5,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,63,64,65,66,67,68,69,70,71,72,74,75,76,77,78,80,81,82,83,84,85,86,87,88,89,90,91,92,94,95,97,98,100,101,102,103,104,105,106,107,108,109,110,111,112],:]
#data2=dataset[[2,3,4,6,13,34,36,60,61,62,73,79,93,99],:]
X_d1=dataset[0:93,1:50]
Y_d1=dataset[0:93,50]
X_d2=dataset[93:,1:50]
Y_d2=dataset[93:,50]
#dataset[:,1:52]=se
#dataset[:,52]=Y
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc = StandardScaler()
#features=scale(features)

X1 = sc.fit_transform(X_d1)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
classifier = SVC(kernel = 'sigmoid', C=0.1,random_state = None, probability=False,gamma='auto')
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score,LeaveOneOut
loocv = LeaveOneOut()
from sklearn.metrics import confusion_matrix, accuracy_score
'''
acc_train=[]
acc_test=[]
for l in range(1):
    X = X1[:,[28,29,30,31]]
    train=[]
    test=[]
    n,m=np.shape(X)
    for count in range(m):
        Select_X=[]
        for j in range(m):
            cm_M=np.zeros((2,2))
            if j!=count:
                Select_X.append(X[:,[j]])
        features = np.concatenate(Select_X,axis=1)
        for k in range(200):
            X_train, X_test, y_train, y_test = train_test_split(features, Y_d1, test_size = 0.05, random_state=None, stratify=Y_d1)
            results = cross_val_score(classifier, X=X_train, y=y_train, cv=loocv)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            test_acc=accuracy_score(y_test,y_pred)
            train.append(np.mean(results))
            test.append(test_acc)
        acc_train.append(np.mean(train))
        acc_test.append(np.mean(test))
    A=np.array(acc_train)


print('train accuracies:',np.array(train))
print('test accuracies:',np.array(test))
np.mean(np.array(train))
'''
X = X1[:,[28,29,30,31]]
train=[]
test=[]
cm_M=np.zeros((2,2))
for i in range(300):
    X_train, X_test, y_train, y_test = train_test_split(X, Y_d1, test_size = 0.15, random_state=None, stratify=Y_d1)
    loocv = LeaveOneOut()
    results = cross_val_score(classifier, X=X_train, y=y_train, cv=loocv)
    #print("LeaveOneOut Accuracy: %.3f%% " % (results.mean()*100.0))
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    test_acc=accuracy_score(y_test,y_pred)
    train.append(np.mean(results))
    test.append(test_acc)
    cm_M=cm_M+cm
    
print('85% Train Data Mean accuracy for 300 runs',np.mean(np.array(train)))

print('15% Test Data Mean accuracy for 300 runs',np.mean(np.array(test)))

print('Confusion Matrix',cm_M/100)
