# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 18:41:43 2019

@author: rayyan
"""



import numpy as np
import pandas as pd
from scipy import stats


# Importing the dataset
dataset = pd.read_csv('Book1_final_algo3.csv')

dataset=dataset.convert_objects(convert_numeric=True)
dataset=dataset.iloc[:,:].values
#Preprocessing missing data fill up using pearson correlation and then with following process

'''
#coorelation
C=dataset[:,[0,1,5,6,7,8,14,29,30,47,51,52]]
Corr=np.corrcoef(C, rowvar=True)
'''
num_binary = dataset[:,[1,5,6,7,8,9,10,11,12,13,14,15,16,18,21,22,23,24,25,26,30,31,38,39,40,41,42,43,44,45,46,47,49,50,51]]
num_continuous = dataset[:,[2,3,4,17,19,20,27,28,29,32,33,34,35,36,37,48]]
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
dataset[:,1:52]=se
################## Train/Test dataset #######################
#data1=dataset[[0,1,5,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,35,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,63,64,65,66,67,68,69,70,71,72,74,75,76,77,78,80,81,82,83,84,85,86,87,88,89,90,91,92,94,95,97,98,100,101,102,103,104,105,106,107,108,109,110,111,112],:]
#data2=dataset[[2,3,4,6,13,34,36,60,61,62,73,79,93,99],:]
X_d1=dataset[0:93,1:52]
Y_d1=dataset[0:93,52]
X_d2=dataset[93:,1:52]
Y_d2=dataset[93:,52]
#dataset[:,1:52]=se
#dataset[:,52]=Y

#############################################################
num_binary = X_d1[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]]
num_multinominal=X_d1[:,[1,14,15,16]]

#############################################################
num_binary1 = num_binary[[0,1,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,25,26,27,28,30,31,32,35,37,38,39,40,41,48,52,53,54,55,56,57,58,60,61,62,64,65,66,67,68,69,70,71,72,74,75,78,82,85],:]
num_binary0 = num_binary[[2,3,4,5,6,7,21,24,29,33,34,36,42,43,44,45,46,47,49,50,51,59,63,73,76,77,79,80,81,83,84,86,87,88,89,90,91,92],:]
#Binary data test
n,m=np.shape(num_binary0)
alist=[]
for num in range(m):
    P=(np.sum(num_binary0[:,num])+np.sum(num_binary1[:,num]))/(len(num_binary0[:,num])+len(num_binary1[:,num]))
    if (P*(len(num_binary0[:,num]))>=5 and P*(len(num_binary1[:,num]))>=5 and (len(num_binary0[:,num]))*(1-P)>=5 and (len(num_binary1[:,num]))*(1-P)>=5):
        Z= ((np.sum(num_binary0[:,num])/len(num_binary0[:,num]))-(np.sum(num_binary1[:,num])/len(num_binary1[:,num])))/np.sqrt(P*(1-P)*((1/len(num_binary0[:,num]))+(1/len(num_binary1[:,num]))))
        p_values = stats.norm.sf(abs(Z))*2
        if (p_values < 0.05):
            alist.append(num_binary[:,[num]])
Bi_features = np.concatenate(alist,axis=1)            

num_multinominal1 = num_multinominal[[0,1,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,25,26,27,28,30,31,32,35,37,38,39,40,41,48,52,53,54,55,56,57,58,60,61,62,64,65,66,67,68,69,70,71,72,74,75,78,82,85],:]
num_multinominal0 = num_multinominal[[2,3,4,5,6,7,21,24,29,33,34,36,42,43,44,45,46,47,49,50,51,59,63,73,76,77,79,80,81,83,84,86,87,88,89,90,91,92],:]

alist=[]
from sklearn.feature_selection import SelectFdr, chi2
n,m=np.shape(num_multinominal)
for num in range(m):
    A1=np.array(num_multinominal1[:,num],dtype=int)
    A0=np.array(num_multinominal0[:,num],dtype=int)
    C1=np.bincount(A1)
    C0=np.bincount(A0)
    if len(C1)-len(C0)>=0:
        C0=np.pad(C0, (0,abs(len(C1)-len(C0))), 'constant')
    if len(C1)-len(C0)<0:
        C1=np.pad(C1, (0,abs(len(C1)-len(C0))), 'constant')
    C=np.array([[C1], [C0]])
    C=C[C != 0]
    chi2_stat, p_val, dof, ex = stats.chi2_contingency(C)
    if p_val<0.05:
        alist.append(num_multinominal[:,[num]])
if len(alist)>0:
    mul_features = np.concatenate(alist, axis=1)




num_continuous = X_d1[:,[35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]]
num_continuous1 = num_continuous[[0,1,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,25,26,27,28,30,31,32,35,37,38,39,40,41,48,52,53,54,55,56,57,58,60,61,62,64,65,66,67,68,69,70,71,72,74,75,78,82,85],:]
num_continuous0 = num_continuous[[2,3,4,5,6,7,21,24,29,33,34,36,42,43,44,45,46,47,49,50,51,59,63,73,76,77,79,80,81,83,84,86,87,88,89,90,91,92],:]
#features selection t-test for Continuous data
n,m=np.shape(num_continuous)
alist=[]
for number in range(m):
    I,P=stats.ttest_ind(num_continuous0[:,number],num_continuous1[:,number],equal_var=False)
    if P<0.05:
        alist.append(num_continuous[:,[number]]) 
con_features = np.concatenate(alist, axis=1)



###############################################################################
features = np.concatenate((Bi_features, con_features), axis=1)
#features = np.concatenate((num_binary, num_multinominal, num_continuous), axis=1)
# Feature Scaling

from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = StandardScaler()
s = MinMaxScaler(feature_range=(0, 1))
#features=scale(features)
features = sc.fit_transform(features)
#LDA supervised extraction model based on dependent variables y................
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
lda = LDA(n_components =None)
features = lda.fit_transform(features,Y_d1)
'''
############################### Algorithm

from sklearn.model_selection import train_test_split
'''
X_train, X_test, y_train, y_test = train_test_split(features, Y_d1, test_size = 0)
'''

#Nonlinear Support Vector Machine..............................................
from sklearn.svm import SVC
classifier = SVC(kernel = 'sigmoid', C=0.1,random_state = None, probability=False,gamma='auto')


from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score,LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score
'''
################################
alist=[]
#K-fold Cross validation
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=None)
for train_index, test_index in rskf.split(X_train, y_train):
    X_trn, X_tst = X_train[train_index], X_train[test_index]
    y_trn, y_tst = y_train[train_index], y_train[test_index]
    print("TRAIN:", y_trn, "TEST:", y_tst)
    TRN=classifier.fit(X_trn, y_trn)
    y_pred = TRN.predict(X_tst)
    cm = confusion_matrix(y_tst, y_pred)
    print("Confusion Matrix: \n ",cm)
    test_acc=accuracy_score(y_tst,y_pred)
    print("1-fold Accuracy: %.3f%% " % (test_acc*100.0))
    alist.append(test_acc)
A=np.array(alist)
k1=np.mean(A[0:5,])
print("10-fold Accuracy: %.3f%% " % (np.mean(A[0:10,])*100.0))
k2=np.mean(A[5:10,])
print("10-fold Accuracy: %.3f%% " % (np.mean(A[10:20,])*100.0))
k3=np.mean(A[10:15,])
print("10-fold Accuracy: %.3f%% " % (np.mean(A[20:30,])*100.0))
k4=np.mean(A[15:20,])
print("10-fold Accuracy: %.3f%% " % (np.mean(A[30:40,])*100.0))
k5=np.mean(A[20:25,])
print("10-fold Accuracy: %.3f%% " % (np.mean(A[40:50,])*100.0))
k6=np.mean(A[25:30,])
print("10-fold Accuracy: %.3f%% " % (np.mean(A[50:60,])*100.0))
k7=np.mean(A[30:35,])
print("10-fold Accuracy: %.3f%% " % (np.mean(A[60:70,])*100.0))
k8=np.mean(A[35:40,])
print("10-fold Accuracy: %.3f%% " % (np.mean(A[70:80,])*100.0))
k9=np.mean(A[40:45,])
print("10-fold Accuracy: %.3f%% " % (np.mean(A[80:90,])*100.0))
k10=np.mean(A[45:50,])
print("10-fold Accuracy: %.3f%% " % (np.mean(A[90:100,])*100.0))

print("Mean Accuracy: %.3f%% " % (A.mean()*100.0))

################################ 
#Leave One Out Crossvalidation
loocv = LeaveOneOut()
results = cross_val_score(classifier, X=X_train, y=y_train, cv=loocv)
print("LeaveOneOut Accuracy: %.3f%% " % (results.mean()*100.0))
'''   
'''

S=classifier.fit(X_train, y_train)
y_pred = TRN.predict(X_train)
cm2 = confusion_matrix(y_train, y_pred)
'''
train=[]
test=[]
cm_M=np.zeros((2,2))
for i in range(300):
    X_train, X_test, y_train, y_test = train_test_split(features, Y_d1, test_size = 0.05, random_state=None, stratify=Y_d1)
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
    
print('90% Train Data Mean accuracy for 100 runs',np.mean(np.array(train)))

print('10% Test Data Mean accuracy for 100 runs',np.mean(np.array(test)))

print('Confusion Matrix',cm_M/100)

