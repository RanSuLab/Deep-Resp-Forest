
# -*- coding: utf-8 -*-

import itertools

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import sklearn.metrics as metrics

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
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








def numeric(target):
    target = pd.to_numeric(target)
    return target
def normalize01(target):
    target = pd.to_numeric(target)
    # print(target)
    target_min = target.min()
    target_max = target.max()


    target_normal = (target-target_min)/(target_max-target_min)
    # print("min:",target_normal.min())
    # print("max:",target_normal.max())
    # print("target_normal:",target_normal)
    return target_normal
def normalize(target):
    #z-score 标准化
    # print(target)
    mean = target.mean()
    std = target.std()
    # print("mean:",mean," std:",std)
    target_normal = (target-mean)/std
    # print("target_normal:",target_normal)
    # print("mean:",target_normal.mean())
    # print("std:",target_normal.std())
    return target_normal
###1:resistant 0:sensitive 2:none
def labeldata(drugdata,target_name):
    # print(drugdata.shape)
    label_y= pd.Series(range(0,drugdata.shape[0]))
    # print(label_y)
    for i in range(0,drugdata.shape[0]):
      if drugdata[target_name][i] >0.8:
          label_y[i] = 1
      elif drugdata[target_name][i] <-0.8:
          label_y[i] = 0
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
    ########################################drug response part#################################################
    # read drug response file
    cgp_drugdata = pd.DataFrame(pd.read_csv('drug.csv'))

    # print(cgp_drugdata.columns)
    cgp_drugdata.info()
    # normalize ic50
    ic50 = cgp_drugdata["LN_IC50"]
    # print(type(ic50))
    ic50_normal = normalize(ic50)
    cgp_drugdata["ic50_normal"]=ic50_normal



    # label (sensitive or resistant)according to ic50_normal
    cgp_drugdata = labeldata(cgp_drugdata,"ic50_normal")


    # order ccle_drugdata by ic50_normal
    cgp_drugdata=cgp_drugdata.sort_values(by="ic50_normal", ascending=False)
    # cgp_drugdata["ic50_normal"].plot(kind="bar")
    # plt.show()
    #
    # output
    # cgp_drugdata.to_csv('cgp_drugdata.csv')
    #
    cgp_drugdata2 = cgp_drugdata[cgp_drugdata["label_y"]!=2]
    cgp_drugdata2.to_csv('cgp_drugdata2.csv')
    drug_for1 = np.unique(cgp_drugdata[cgp_drugdata["label_y"]==1]["DRUG_ID"])
    drug_for0 = np.unique(cgp_drugdata[cgp_drugdata["label_y"]==0]["DRUG_ID"])
    drugs = [val for val in list(drug_for1) if val in list(drug_for0)]
    print(drugs)


    ########################################gene exprSet part#################################################
    # read gene exprset file
    L=[]
    file=open("gene.txt","r")
    line=file.readline()
    while line:
        line=line.split()
        L.append(line)
        line=file.readline()
    cgp_exprSet = pd.DataFrame(L)

    cgp_exprSet2=cgp_exprSet.T
    # print(cgp_exprSet2)
    cgp_exprSet3 = cgp_exprSet2.drop([0]).reset_index(drop=True)
    cgp_exprSet3.columns=cgp_exprSet2.iloc[0,:].tolist()
    # print(cgp_exprSet3)
    # cgp_exprSet3.to_csv('cgp_exprSet3.csv')
    cgp_exprSet4 = pd.DataFrame(index=cgp_exprSet3.index)
    cgp_exprSet4[cgp_exprSet3.columns[0]]=cgp_exprSet3.iloc[:,0]
    for i in range(1,17738):
        each_column = cgp_exprSet3[cgp_exprSet3.columns[i]]
        cgp_exprSet4[cgp_exprSet3.columns[i]] = normalize01(each_column)
    # print(cgp_exprSet4)
    cgp_exprSet4.info()
    # cgp_exprSet4.to_csv('cgp_exprSet4.csv')






    ########################################copy number part#################################################
    # read copy number file
    L=[]
    file=open("cna.txt","r")
    line=file.readline()
    while line:
        line=line.split()
        L.append(line)
        line=file.readline()
    cgp_cnaSet = pd.DataFrame(L)
    cgp_cnaSet2 = cgp_cnaSet.drop([0]).reset_index(drop=True)
    cgp_cnaSet2.columns=cgp_cnaSet.iloc[0,:].tolist()
    # print(cgp_cnaSet2)
    # cgp_cnaSet2.info()
    cgp_cnaSet3 = pd.DataFrame(index=cgp_cnaSet2.index)
    cgp_cnaSet3[cgp_cnaSet2.columns[0]]=cgp_cnaSet2.iloc[:,0]
    for i in range(1,426):
        each_column = cgp_cnaSet2[cgp_cnaSet2.columns[i]]
        cgp_cnaSet3[cgp_cnaSet2.columns[i]] = numeric(each_column)
    cgp_cnaSet3.info()
    # cgp_cnaSet3.to_csv('cgp_cnaSet3.csv')








    ##################################generate x and y for each drug######################################
    drugs = [1,5,29,32,34,37,38,41,45,52,55,56,62,71,88]
    # for i in cgp_drugdata["DRUG_ID"].tolist():
    #   if i not in drugs:
    #     drugs.append(i)
    # print(drugs)

    #####choose the same celllines in both gene and cna
    cgp_exprSet4.rename(columns={"ensembl_gene":"COSMIC_ID"},inplace=True)##inplace=True表示在原数据上进行操作
    cgp_cnaSet3.rename(columns={"Name":"COSMIC_ID"},inplace=True)
    exprCName = cgp_exprSet4["COSMIC_ID"]
    # print(list(exprCName))
    cnaCName = cgp_cnaSet3["COSMIC_ID"]
    # print(list(cnaCName))
    conCName = [val for val in list(exprCName) if val in list(cnaCName)]
    conCName.remove("905954")
    conCName.remove("905954")
    conCName.remove("909976")
    conCName.remove("909976")
    conCName.remove("1330983")
    conCName.remove("1330983")
    conCName.remove("1503362")
    conCName.remove("1503362")
    print("con-cellline:",list(conCName))
    # pd.DataFrame(list(conCName)).to_csv("conCName.csv")


    #####for each drug
    all_drug_acc=[]
    for i in range(15):
        each_drug_acc = []
        drug=drugs[i]
        each_drug_acc.append(drug)
        each_drugdata = pd.DataFrame(cgp_drugdata2[cgp_drugdata2["DRUG_ID"]==drugs[i]])
        # each_drugdata.to_csv('each_drugdata.csv')
        print(each_drugdata)


        ######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~prepare x and y(gene part)~~~~~~~~~~~~~~~~~~~~~~~~~####################
        cgp_exprSet4['COSMIC_ID'] = pd.to_numeric(cgp_exprSet4['COSMIC_ID'])
        cgp_exprSet5 = cgp_exprSet4[cgp_exprSet4['COSMIC_ID'].isin(list(conCName))]
        cgp_exprSet5["COSMIC_ID"].to_csv("cgp_exprSet5.csv")

        each_data = pd.merge(each_drugdata,cgp_exprSet5,on="COSMIC_ID",how="inner")
        print(each_data)
        ###########################################test gcForest######################################33
        random.seed(5)
        randomCols = random.sample(range(10,17747),400)
        x = each_data.iloc[:,randomCols]
        y = each_data.iloc[:,9]
        x.index=each_data["COSMIC_ID"]
        y.index=each_data["COSMIC_ID"]
        x.sort_index()
        y.sort_index()
        print(x)
        print(y)
        folds_expr = five_fold(x.shape[0])
        # pd.DataFrame(folds_expr).to_csv("svm_rf/folds_expr.csv")



        ######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~prepare x and y(cna part)~~~~~~~~~~~~~~~~~~~~~~~~~####################
        cgp_cnaSet3['COSMIC_ID'] = pd.to_numeric(cgp_cnaSet3['COSMIC_ID'])
        cgp_cnaSet4 = cgp_cnaSet3[cgp_cnaSet3['COSMIC_ID'].isin(list(conCName))]
        cgp_cnaSet4["COSMIC_ID"].to_csv("cgp_cnaSet4.csv")
        each_cna_data = pd.merge(each_drugdata,cgp_cnaSet4,on="COSMIC_ID",how="inner")
        ###########################################test gcForest######################################33
        random.seed(5)
        randomCols = random.sample(range(10,435),400)
        x_cna = each_cna_data.iloc[:,randomCols]
        y_cna = each_cna_data.iloc[:,9]
        x_cna.index=each_cna_data["COSMIC_ID"]
        y_cna.index=each_cna_data["COSMIC_ID"]
        x_cna.sort_index()
        y_cna.sort_index()
        folds_cna = five_fold(x_cna.shape[0])
        # pd.DataFrame(folds_cna).to_csv("svm_rf/folds_cna.csv")

        ############################################five fold##########################################
        feature_flag = ""
        five_results = []
        for j in range(5):
            each_fold_result=[]
            each_fold_result.append(j)
            X_test = x.iloc[folds_expr[j],:]
            y_test = y.iloc[folds_expr[j]]
            X_train = x.iloc[list(set(range(x.shape[0])).difference(set(folds_expr[j]))),:]
            y_train = y.iloc[list(set(range(x.shape[0])).difference(set(folds_expr[j])))]
            X_test.to_csv("svm_rf/"+str(j)+str(drug)+"_X_test_expr.csv")
            y_test.to_csv("svm_rf/"+str(j)+str(drug)+"_y_test_expr.csv")

            X_test_cna = x_cna.iloc[folds_cna[j],:]
            y_test_cna = y_cna.iloc[folds_cna[j]]
            X_train_cna = x_cna.iloc[list(set(range(x.shape[0])).difference(set(folds_cna[j]))),:]
            y_train_cna = y_cna.iloc[list(set(range(x.shape[0])).difference(set(folds_cna[j])))]
            X_test_cna.to_csv("svm_rf/"+str(j)+str(drug)+"_X_test_cna.csv")
            y_test_cna.to_csv("svm_rf/"+str(j)+str(drug)+"_y_test_cna.csv")


            ######################mgs expr part################
            levels = np.unique(np.array(y_train))
            print("levels:",levels)
            File = open("svm_rf/"+str(j)+str(drug)+".txt", "w")
            File.write("levels:"+str(levels)+"\n")
            clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr',probability=True)
            clf.fit(np.array(X_train), np.array(y_train))
            print(clf.score(np.array(X_train), np.array(y_train)))  # 精度
            y_svm = clf.predict(np.array(X_test))
            pd.DataFrame(y_svm).to_csv("svm_rf/"+str(j)+str(drug)+"_svm_y_svm.csv")
            prediction_accuracy = accuracy_score(y_true=y_test, y_pred=y_svm)
            print(prediction_accuracy)
            each_fold_result.append(prediction_accuracy)
            print('svm Layer validation accuracy = {}'.format(prediction_accuracy))
            File.write('svm prediction_accuracy = {}'.format(prediction_accuracy)+"\n")
            pred_proba = clf.predict_proba(np.array(X_test))
            pd.DataFrame(pred_proba).to_csv("svm_rf/"+str(j)+str(drug)+"_svm_pred_proba.csv")
            pred = pred_proba[:,1]
            print("pred:",pred)
            fpr, tpr, thresholds = roc_curve(y_test, pred, pos_label=1)
            if fpr.shape[0]==1:
                auc_svm = np.nan
            else:
                auc_svm = auc(fpr, tpr)
            each_fold_result.append(auc_svm)
            print("auc:",auc_svm)
            File.write('svm auc = {}'.format(auc_svm)+"\n")
            File.close()
            ######################rf part################
            File = open("svm_rf/"+str(j)+str(drug)+"rf.txt", "w")
            File.write("levels:"+str(levels)+"\n")
            rf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                                         min_samples_split=0.05, oob_score=True, n_jobs=1,random_state=5)
            rf.fit(np.array(X_train),np.array(y_train))
            pred_proba_rf = rf.predict_proba(X=np.array(X_test))
            pd.DataFrame(pred_proba_rf).to_csv("svm_rf/"+str(j)+str(drug)+"_rf_predict_proba.csv")
            predictions_rf = levels[np.argmax(pred_proba_rf, axis=1)]
            pd.DataFrame(predictions_rf).to_csv("svm_rf/"+str(j)+str(drug)+"_rf_predictions.csv")
            prediction_accuracy_rf = accuracy_score(y_true=y_test, y_pred=predictions_rf)
            each_fold_result.append(prediction_accuracy_rf)
            print('rf Layer validation accuracy = {}'.format(prediction_accuracy_rf))
            File.write('rf prediction_accuracy = {}'.format(prediction_accuracy_rf)+"\n")
            pred_rf = pred_proba_rf[:,1]
            print("pred_rf:",pred_rf)
            fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, pred_rf, pos_label=1)

            auc_rf = auc(fpr_rf, tpr_rf)
            each_fold_result.append(auc_rf)
            print("auc:",auc_rf)
            File.write('rf auc = {}'.format(auc_rf)+"\n")
            File.close()
            five_results.append(each_fold_result)
        pd.DataFrame(five_results).to_csv("svm_rf/"+str(drug)+"five_results.csv")
        print(five_results)
        each_drug_acc.append(pd.DataFrame(five_results).iloc[:,1].mean())
        each_drug_acc.append(pd.DataFrame(five_results).iloc[:,2].mean())
        each_drug_acc.append(pd.DataFrame(five_results).iloc[:,3].mean())
        each_drug_acc.append(pd.DataFrame(five_results).iloc[:,4].mean())
        all_drug_acc.append(each_drug_acc)
    pd.DataFrame(all_drug_acc).to_csv("svm_rf/all_drug_acc.csv")



