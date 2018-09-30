import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.svm import SVC
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

ALLdata = pd.read_csv('CleanAllVectors.txt', header=None, skiprows=0, delim_whitespace=True, quoting=csv.QUOTE_NONE, error_bad_lines=False)
ALLdata.columns = ['word', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100']
print(ALLdata.shape)

ALLdata['label'] = 0
print(ALLdata.shape)

dis = pd.read_csv('DiseasVectors.txt', header=None, skiprows=0, delim_whitespace=True)
dis.columns = ['word', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100']
print(dis.shape)

dis['label'] = 1

# ALLdata['label'] = ALLdata.merge(dis[['word']], how='left', indicator=True
#                                    )['_merge'].eq('both').astype(int)
print(dis.shape)

neg = ALLdata.sample(n=80000, replace=False)

po1 = dis.sample(n=4000, replace=False)
po2 = dis.sample(n=4000, replace=False)
po3 = dis.sample(n=4000, replace=False)
po4 = dis.sample(n=4000, replace=False)
po5 = dis.sample(n=4000, replace=False)

po6 = dis.sample(n=4000, replace=False)
po7 = dis.sample(n=4000, replace=False)
po8 = dis.sample(n=4000, replace=False)
po9 = dis.sample(n=4000, replace=False)
po10 = dis.sample(n=4000, replace=False)

po11 = dis.sample(n=4000, replace=False)
po12 = dis.sample(n=4000, replace=False)
po13 = dis.sample(n=4000, replace=False)
po14 = dis.sample(n=4000, replace=False)

po15 = dis.sample(n=4000, replace=False)
po16 = dis.sample(n=4000, replace=False)
po17 = dis.sample(n=4000, replace=False)
po18 = dis.sample(n=4000, replace=False)
po19 = dis.sample(n=4000, replace=False)
po20 = dis.sample(n=4000, replace=False)

Allpositive = pd.concat([po1, po2, po3, po4, po5, po6, po7, po8, po9, po10, po11, po12,po13,po14,po15,po16,po17,po18,po19,po20], ignore_index=True)
print(Allpositive.shape)

Alldata = pd.concat([Allpositive, neg]).sort_index(kind='merge')
print(Alldata.shape)

Alldata.to_csv('2resbinary.txt', sep='\t')

targetdata = Alldata['label'].values.ravel()
print(targetdata.shape)
data = Alldata[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100']]
print(data.shape)
data = np.array(data.values)
# Alldata = np.array(Alldata.values)
#
X_train, X_test, y_train, y_test = train_test_split(data, targetdata, test_size=0.2, random_state=0)

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')

# fold = 10
# skf = StratifiedShuffleSplit(n_splits=fold)
# f1 = 0.0
# for train_index, test_index in skf.split(data, targetdata):
#     X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
#     clf.fit(X_train, y_train)
#     # predicted = clf.predict(X_test)
#     # predicted1 = clf.predict(X_train)
#     # pro = clf.predict_proba(X_test)
#     # pro1 = clf.predict_proba(X_train)
#     # pro = clf.predict_log_proba(X_test)
#     # np.savetxt('testscore.tsv', pro, delimiter="\t")
#     # np.savetxt('trainscore.tsv', pro1, delimiter="\t")
#     # temp = Alldata[test_index,0:2]
#     # temp1 = Alldata[train_index, 0:2]
#     # temp.to_csv('file2.tsv', sep='\t')
#     # predictiontemp = pd.DataFrame(temp, columns=['word'])
#     # predictiontemp1 = pd.DataFrame(temp1, columns=['word'])
#     # predictiontemp.to_csv('testword.tsv', sep='\t')
#     # predictiontemp1.to_csv('trainword.tsv', sep='\t')
#     predicted = clf.predict(X_test)
#     f1 += f1_score(y_test, predicted, average='micro')
#
# # Calculate AUC
# fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted)
# aus_score = metrics.roc_auc_score(y_test, predicted)
# plt.plot(fpr,tpr,label='ROC curve for Linear SVM (area = %0.4f)' % aus_score)
# print(f1 / fold)
#
mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=10, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = StratifiedShuffleSplit(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

# Calculate AUC
fpr, tpr, _ = metrics.roc_curve(y_test.ravel(), predicted.ravel())
aus_score = metrics.roc_auc_score(y_test, predicted, average='macro')
print(aus_score)
plt.plot(fpr,tpr,label='ROC curve for 10 ANN (area = %0.4f)' % aus_score)
# print(f1 / fold)
#
mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=50, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = StratifiedShuffleSplit(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

# Calculate AUC
fpr, tpr, _ = metrics.roc_curve(y_test.ravel(), predicted.ravel())
aus_score = metrics.roc_auc_score(y_test, predicted, average='macro')
print(aus_score)
plt.plot(fpr,tpr,label='ROC curve for 50 ANN (area = %0.4f)' % aus_score)
# print(f1 / fold)
#
mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=100, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = StratifiedShuffleSplit(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

# Calculate AUC
fpr, tpr, _ = metrics.roc_curve(y_test.ravel(), predicted.ravel())
aus_score = metrics.roc_auc_score(y_test, predicted, average='macro')
print(aus_score)
plt.plot(fpr,tpr,label='ROC curve for 100 ANN (area = %0.4f)' % aus_score)
# print(f1 / fold)

mlp = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=200, activation='logistic', alpha=0.1, shuffle=True, solver="lbfgs"))
fold = 10
skf = StratifiedShuffleSplit(n_splits=fold)
f1 = 0.0
for train_index, test_index in skf.split(data, targetdata):
    X_train, X_test, y_train, y_test = data[train_index, :], data[test_index, :], targetdata[train_index], targetdata[test_index]
    clf = mlp
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    f1 += f1_score(y_test, predicted, average='micro')

# Calculate AUC
fpr, tpr, _ = metrics.roc_curve(y_test.ravel(), predicted.ravel())
aus_score = metrics.roc_auc_score(y_test, predicted, average='micro')
print(aus_score)
plt.plot(fpr,tpr,label='ROC curve for 200 ANN (area = %0.4f)' % aus_score)
# print(f1 / fold)

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve - Diseases")
plt.legend(loc="best")
plt.show()