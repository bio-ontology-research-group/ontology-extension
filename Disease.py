import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

ALLdata = pd.read_csv('CleanAllVectors.txt', header = None, skiprows = 0, sep = ' ', quoting=csv.QUOTE_NONE, error_bad_lines=False)
print(ALLdata.shape)

Fdata = ALLdata[0]
print(Fdata.shape)


Redata = ALLdata.iloc[:, 1:101]
print(Redata.shape)

Redata.insert(100, '', 0)
Redata[100] = 0
print(Redata.shape)

newF = Fdata.fillna('0')

count = 0
for i in newF:
     if 'doid' in i:
         Redata.set_value(count, 100, 1)
         count=count+1
     else:
         count=count+1
print(Redata.shape)

# count = 0
# for i in newF:
#      if '?' or ']' or '[' or ',' or '.' or '0' in i:
#          Redata.drop(Redata.index[[count]])
#          count=count+1
#          print(count)
#      else:
#          count=count+1
#          print(count)
# # #
# print(Redata.shape)


Dwdata = Redata[Redata[100] == 1]
NonDwdata = Redata[Redata[100] == 0]
print(Dwdata.shape)
print(NonDwdata.shape)


RDsample = Dwdata.sample(n=4000, replace=False)
RNDsample = NonDwdata.sample(40000, replace=False)



combineA = RDsample.append(RNDsample)
print(combineA.shape)

data = combineA.iloc[:, 0:100]
print(data.shape)
targetdata = combineA[100]
print(targetdata.shape)

X_train, X_test, y_train, y_test = train_test_split(data, targetdata, test_size=0.2, random_state=0)
# To know the number of training and test samples
# w1 = y_train[y_train == 1]
# w2 = y_train[y_train == 0]
# print(w1.shape)
# print(w2.shape)
#
#
# w1 = y_test[y_test == 1]
# w2 = y_test[y_test == 0]
# print(w1.shape)
# print(w2.shape)
data = np.array(data.values)
targetdata = targetdata.fillna('0') 

fold = 10
skf = KFold(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

print(f1 / fold)

mlp = MLPClassifier(hidden_layer_sizes=10, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs")
fold = 10
skf = KFold(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

print(f1 / fold)

mlp1 = MLPClassifier(hidden_layer_sizes=25, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs")
fold = 10
skf = KFold(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp1
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

print(f1 / fold)

mlp2 = MLPClassifier(hidden_layer_sizes=50, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs")
fold = 10
skf = KFold(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp2
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

print(f1 / fold)

mlp3 = MLPClassifier(hidden_layer_sizes=100, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs")
fold = 10
skf = KFold(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp3
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

print(f1 / fold)

mlp4 = MLPClassifier(hidden_layer_sizes=200, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs")
fold = 10
skf = KFold(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp4
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

print(f1 / fold)

# Compute ROC curve and ROC area for each class
# print(n_classes)
# print(y_test.shape)
# print(y_score.shape)
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
#
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score)
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
# # plot it for each class
# plt.figure()
# lw = 2
# plt.plot(fpr[0], tpr[0], color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic for NonDiseas')
# plt.legend(loc="lower right")
# plt.show()

# for Tsne

# tsne = TSNE(n_components=4, random_state=0)
# X_2d = tsne.fit_transform(data)
#
# target_ids = range(2)
#
# plt.figure(figsize=(6, 5))
# colors = 'orange', 'purple'
# for i, c, label in zip(target_ids, colors, ['Non-Disease','Disease']):
#     plt.scatter(X_2d[targetdata == i, 0], X_2d[targetdata == i, 1], c=c, label=label)
# plt.legend()
# plt.show()