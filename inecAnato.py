import pandas as pd
import numpy as np
import csv
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import interp
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# dis = pd.read_csv('DiseasVectors.txt', header=None, skiprows=0, delim_whitespace=True)
# dis.columns = ['Doid', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100']
# print(dis.shape)
#
# # For labeling
#
# d1 = pd.read_csv('t1out.txt', sep = '\t')
# d2 = pd.read_csv('t2out.txt', sep = '\t')
# d3 = pd.read_csv('t3out.txt', sep = '\t')
# d4 = pd.read_csv('t4out.txt', sep = '\t')
# d5 = pd.read_csv('t5out.txt', sep = '\t')
# d6 = pd.read_csv('t6out.txt', sep = '\t')
# d7 = pd.read_csv('t7out.txt', sep = '\t')
# d8 = pd.read_csv('t8out.txt', sep = '\t')
# d9 = pd.read_csv('t9out.txt', sep = '\t')
# d10 = pd.read_csv('t10out.txt', sep = '\t')
# d11 = pd.read_csv('t11out.txt', sep = '\t')
# d12= pd.read_csv('t12out.txt', sep = '\t')
# d13= pd.read_csv('t13out.txt', sep = '\t')
# d14= pd.read_csv('t14out.txt', sep = '\t')
# d15= pd.read_csv('t15out.txt', sep = '\t')
# d16= pd.read_csv('t16out.txt', sep = '\t')
#
# d1.columns = ['Doid','label1']
# d2.columns = ['Doid','label2']
# d3.columns = ['Doid','label3']
# d4.columns = ['Doid','label4']
# d5.columns = ['Doid','label5']
# d6.columns = ['Doid','label6']
# d7.columns = ['Doid','label7']
# d8.columns = ['Doid','label8']
# d9.columns = ['Doid','label9']
# d10.columns = ['Doid','label10']
# d11.columns = ['Doid','label11']
# d12.columns = ['Doid','label12']
# d13.columns = ['Doid','label13']
# d14.columns = ['Doid','label14']
# d15.columns = ['Doid','label15']
# d16.columns = ['Doid','label16']
#
#
# result = dis.merge(d1,how='left', on=['Doid'])
#
# result = result.merge(d2,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label2'])
#
# result = result.merge(d3,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label3'])
#
# result = result.merge(d4,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label4'])
#
# result = result.merge(d5,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label5'])
#
# result = result.merge(d6,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label6'])
#
# result = result.merge(d7,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label7'])
#
# result = result.merge(d8,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label8'])
#
# result = result.merge(d9,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label9'])
#
# result = result.merge(d10,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label10'])
#
# result = result.merge(d11,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label11'])
#
# result = result.merge(d12,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label12'])
#
# result = result.merge(d13,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label13'])
#
# result = result.merge(d14,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label14'])
#
# result = result.merge(d15,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label15'])
#
# result = result.merge(d16,how='left', on=['Doid'])
#
# result['label1'] = result['label1'].replace('',np.NaN).fillna(result['label16'])
#
# print(result.shape)
#
# result.to_csv('2resboth.txt', sep='\t')

disvec = pd.read_csv('2resboth.txt', header=None, skiprows=0, delim_whitespace=True)
disvec.columns = ['doid', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','label']
print(disvec.shape)

targetdata = disvec['label'].values.ravel()
print(targetdata.shape)
# Here you can determine the classes you want to test against.
# 0,1,2,3,4 -> For infectious disease experiment
# 0,5,6,7,8,9,10,11,12,13,14,15,16 -> For anatomical diseases experiment
# 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 -> For Combining infectious and anatomical diseases experiment
targetdata = preprocessing.label_binarize(targetdata, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]) #,5,6,7,8,9,10,11,12,13,14,15,16
data = disvec[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100']]
print(data.shape)
data = np.array(data.values)
disvec = np.array(disvec.values)
X_train, X_test, y_train, y_test = train_test_split(data, targetdata, test_size=0.2, random_state=0)

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')

mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=10, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = StratifiedShuffleSplit(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    predict_probabilities = clf.predict_proba(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

# Calculate AUC
print(y_test.shape)
print(predict_probabilities.shape)
fpr, tpr, thresholds = metrics.roc_curve(y_test.ravel(), predict_probabilities.ravel())
aus_score = metrics.roc_auc_score(y_test, predicted, average='macro')
print(aus_score)
plt.plot(fpr,tpr,label='ROC curve for 10 neurons ANN (area = %0.4f)' % aus_score)
print(f1 / fold)

mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=50, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = StratifiedShuffleSplit(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    predict_probabilities = clf.predict_proba(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

# Calculate AUC
fpr, tpr, _ = metrics.roc_curve(y_test.ravel(), predict_probabilities.ravel())
aus_score = metrics.roc_auc_score(y_test, predicted, average='macro')
print(aus_score)
plt.plot(fpr,tpr,label='ROC curve for 50 neurons ANN (area = %0.4f)' % aus_score)
print(f1 / fold)

mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=100, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = StratifiedShuffleSplit(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    predict_probabilities = clf.predict_proba(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

# Calculate AUC
fpr, tpr, _ = metrics.roc_curve(y_test.ravel(), predict_probabilities.ravel())
aus_score = metrics.roc_auc_score(y_test, predicted, average='macro')
print(aus_score)
plt.plot(fpr,tpr,label='ROC curve for 100 neurons ANN (area = %0.4f)' % aus_score)
print(f1 / fold)

mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=200, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = StratifiedShuffleSplit(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    predict_probabilities = clf.predict_proba(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

# Calculate AUC
fpr, tpr, _ = metrics.roc_curve(y_test.ravel(), predict_probabilities.ravel())
aus_score = metrics.roc_auc_score(y_test, predicted, average='micro')
print(aus_score)
plt.plot(fpr,tpr,label='ROC curve for 200 neurons ANN (area = %0.4f)' % aus_score)
print(f1 / fold)

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve - Combine infectious and anatomical diseases")
plt.legend(loc="best", prop={'size': 9})
plt.show()
