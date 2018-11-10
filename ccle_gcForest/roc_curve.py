
# -*- coding: utf-8 -*-

import itertools

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt



import sklearn.metrics as metrics

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split



from GCForest_fs import gcForest
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc



def plot_confusion_matrix(cm, classes, normalize=False,

                          title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()





def gcf(X_train, X_test, y_train, y_test, cnames):



    clf = gcForest(shape_1X=(1, 18988),window=[1000,2000],stride=10)

    clf.fit(X_train, y_train)



    y_pred = clf.predict(X_test)
    print(y_pred)




   # print('accuracy:', metrics.accuracy_score(y_test, y_pred))

    #print('kappa:', metrics.cohen_kappa_score(y_test, y_pred))

    #print(metrics.classification_report(y_test, y_pred, target_names=cnames))



    #cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    #plot_confusion_matrix(cnf_matrix, classes=cnames, normalize=True,

    #                  title='Normalized confusion matrix')
def normalize01(target):
    target = pd.to_numeric(target)
    # print(target)
    target_min = target.min()
    target_max = target.max()


    target_normal = (target-target_min)/(target_max-target_min)
    return target_normal
def normalize(target):
    #z-score 标准化
    # print(target)
    mean = target.mean()
    std = target.std()
    print("mean:",mean," std:",std)
    target_normal = (target-mean)/std
    # print("target_normal:",target_normal)
    # print("max:",target_normal.max())
    # print("min:",target_normal.min())
    return target_normal

def labeldata(drugdata,target_name):
    # print(drugdata.shape)
    label_y= pd.Series(range(0,drugdata.shape[0]))
    # print(label_y)
    for i in range(0,drugdata.shape[0]):
      if drugdata[target_name][i] >0.8:
          label_y[i] = 0
      elif drugdata[target_name][i] <-0.8:
          label_y[i] = 1
      else:
          label_y[i] = 2
    # print(label_y)
    drugdata["label_y"] = label_y
    return drugdata


def five_fold(n):
    L=[]
    np.random.seed(5)
    numbers = np.random.permutation(range(n))
    print(numbers)
    if n==8 :
        a1 = numbers[0:2]
        L.append(a1)
        a2 = numbers[2:4]
        L.append(a2)
        a3 = numbers[4:6]
        L.append(a3)
        a4 = numbers[6:7]
        L.append(a4)
        a5 = numbers[7:]
        L.append(a5)
    else:
        a1 = numbers[0:round(n/5)]
        L.append(a1)
        a2 = numbers[round(n/5):2*round(n/5)]
        L.append(a2)
        a3 = numbers[2*round(n/5):3*round(n/5)]
        L.append(a3)
        a4 = numbers[3*round(n/5):4*round(n/5)]
        L.append(a4)
        a5 = numbers[4*round(n/5):]
        L.append(a5)
    return L


if __name__ == '__main__':

    drugs=["AEW541","Erlotinib"]
    #####for each drug
    for i in range(1,2):
        for j in range(1):
            y_test = pd.read_csv("single/"+str(j)+str(drugs[i])+"_y_test_expr.csv",header = None)
            print(y_test)
            predict_proba = pd.read_csv("single/"+str(j)+str(drugs[i])+"_predict_proba.csv")
            print(predict_proba.iloc[:,2])
            fpr,tpr,threshold = roc_curve(y_test.iloc[:,1], predict_proba.iloc[:,2]) ###计算真正率和假正率
            roc_auc = auc(fpr,tpr) ###计算auc的值

            y_test_four = pd.read_csv("four/"+str(j)+str(drugs[i])+"_y_test_expr.csv",header = None)
            print(y_test_four)
            predict_proba_four = pd.read_csv("four/"+str(j)+str(drugs[i])+"_predict_proba.csv")
            print(predict_proba_four.iloc[:,2])
            fpr_four,tpr_four,threshold_four = roc_curve(y_test_four.iloc[:,1], predict_proba_four.iloc[:,2]) ###计算真正率和假正率
            roc_auc_four = auc(fpr_four,tpr_four) ###计算auc的值



            # 设置刻度字体大小
            plt.figure(figsize=(10,10))
            plt.rc('font',family='Times New Roman',weight = "ultralight",size=14)
            plt.xticks()
            plt.yticks()
            lw = 2
            plt.plot(fpr_four, tpr_four, color='red',lw=lw, label='MIMGS 1 (Area = %0.2f)' % roc_auc_four)
            plt.plot(fpr, tpr, color='green',lw=lw, label='MIMGS 2 (Area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="right",frameon=False)
            plt.text(0.9,0.05,"AEW541")
            plt.show()



            # tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
            # specificity = tn /float(tn+fp)
            # each_fold_result.append(specificity)
            # print("specificity:",specificity)
            # File.write('specificity = {}'.format(specificity)+"\n")
            # sensitivity= tp/float(tp+fn)
            # each_fold_result.append(sensitivity)
            # print("sensitivity:",sensitivity)
            # File.write('sensitivity = {}'.format(sensitivity)+"\n")














