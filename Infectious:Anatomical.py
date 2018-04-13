#include just the disease without other words.
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
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
ALLdata = pd.read_csv('CleanAllVectors.txt', header = None, skiprows = 0, sep = ' ', quoting=csv.QUOTE_NONE, error_bad_lines=False)
DiseasesData = pd.read_table('DiseasVectors.txt', delim_whitespace=True)

# Upload the Files generated from generateMAP.groovy that corresponding to each subclass of Infectious/Anatomical diseases

d1 = pd.read_csv('t1out.txt', sep = '\t')
d2 = pd.read_csv('t2out.txt', sep = '\t')
d3 = pd.read_csv('t3out.txt', sep = '\t')
d4 = pd.read_csv('t4out.txt', sep = '\t')
d5 = pd.read_csv('t5out.txt', sep = '\t')
d6 = pd.read_csv('t6out.txt', sep = '\t')
d7 = pd.read_csv('t7out.txt', sep = '\t')
d8 = pd.read_csv('t8out.txt', sep = '\t')
d9 = pd.read_csv('t9out.txt', sep = '\t')
d10 = pd.read_csv('t10out.txt', sep = '\t')
d11 = pd.read_csv('t11out.txt', sep = '\t')
d12= pd.read_csv('t12out.txt', sep = '\t')
d13= pd.read_csv('t13out.txt', sep = '\t')
d14= pd.read_csv('t14out.txt', sep = '\t')
d15= pd.read_csv('t15out.txt', sep = '\t')
d16= pd.read_csv('t16out.txt', sep = '\t')
# d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16
# print(DiseasesData.shape)
# print(d1.shape)
d1.columns = ['Doid','class']
d2.columns = ['Doid','class']
d3.columns = ['Doid','class']
d4.columns = ['Doid','class']
d5.columns = ['Doid','class']
d6.columns = ['Doid','class']
d7.columns = ['Doid','class']
d8.columns = ['Doid','class']
d9.columns = ['Doid','class']
d10.columns = ['Doid','class']
d11.columns = ['Doid','class']
d12.columns = ['Doid','class']
d13.columns = ['Doid','class']
d14.columns = ['Doid','class']
d15.columns = ['Doid','class']
d16.columns = ['Doid','class']

DiseasesData.columns = ['Doid', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100']
# DiseasesData['class0'] = 0
#d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16
N= 3669
frames = [d1,d2,d3,d4]
result = pd.concat(frames)
# d1.columns = ['Doid','class']
print(result.shape)
# print(DiseasesData.shape)

DiseasesData['Doid'] = DiseasesData['Doid'].str.lower()
result['Doid'] = result['Doid'].str.lower()

df4 = pd.merge(DiseasesData, result, on='Doid')
print(df4.shape)

DF2 = pd.merge(DiseasesData, df4, how='outer', indicator=True)
DF2 = DF2[DF2._merge == 'left_only'].drop('_merge', axis=1)
print(DF2.shape)
DF2['class'] = 0
ranNonData = DF2.sample(n=N, replace=False)
# ALLdata.columns = ['Doid', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','class']
# ALLdata['class'] = 17
# ranData = ALLdata.sample(n=4132, replace=False)
WholeCombine = df4.append(ranNonData)
# WholeCombine1 = WholeCombine.append(ranData)
print(WholeCombine.shape)

# #################
data = WholeCombine.iloc[:, 1:102]
print(data.shape)
targetdata = WholeCombine['class']
print(targetdata.shape)
targetdata = preprocessing.label_binarize(targetdata, classes=[0,1,2,3,4]) #,5,6,7,8,9,10,11,12,13,14,15,16
# n_classes = targetdata.shape[1]

X_train, X_test, y_train, y_test = train_test_split(data, targetdata, test_size=0.2, random_state=0)

# To know the number of training and test samples for each class
# w1 = y_train[y_train == 1]
# w2 = y_train[y_train == 2]
# w3 = y_train[y_train == 3]
# w4 = y_train[y_train == 4]
# print(w1.shape)
# print(w2.shape)
# print(w3.shape)
# print(w4.shape)
#
# w5 = y_test[y_test == 1]
# w6 = y_test[y_test == 2]
# w7 = y_test[y_test == 3]
# w8 = y_test[y_test == 4]
# print(w5.shape)
# print(w6.shape)
# print(w7.shape)
# print(w8.shape)
# ################
#
results = {}
data = np.array(data.values)
#
#
fold = 10
skf = KFold(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index, :], targetdata[test_index,:]
    clf = OneVsRestClassifier(svm.LinearSVC())
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

print(f1 / fold)

mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=10, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = KFold(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index, :], targetdata[test_index,:]
    clf = mlp
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

print(f1 / fold)

mlp1 = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=25, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = KFold(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index, :], targetdata[test_index,:]
    clf = mlp1
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

print(f1 / fold)

mlp2 = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=50, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = KFold(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index, :], targetdata[test_index,:]
    clf = mlp2
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

print(f1 / fold)

mlp3 = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=100, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = KFold(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index, :], targetdata[test_index,:]
    clf = mlp3
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

print(f1 / fold)

mlp4 = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=200, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = KFold(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index, :], targetdata[test_index,:]
    clf = mlp4
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')
    # Calculate Confusion Matrix
    CM = confusion_matrix(y_test,predicted)

print(f1 / fold)

# For AUC
svmclassi = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=10, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))# Change it each time with ANN (10,25,50,100,200)
# results['NN_params_10'] = cross_val_score(mlp1, data, targetdata, cv = kfold).mean()
y_score = svmclassi.fit(X_train, y_train).decision_function(X_test)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

