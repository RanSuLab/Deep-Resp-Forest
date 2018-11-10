
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
    ########################################drug response part#################################################
    # read drug response file
    ccle_drugdata = pd.DataFrame(pd.read_csv('drug.csv'))
    # print(ccle_drugdata.columns)
    ccle_drugdata.info()
    # normalize actarea
    actarea = ccle_drugdata["ActArea"]
    acta_normal = normalize(actarea)
    ccle_drugdata["acta_normal"]=acta_normal

    # normalize ic50
    ic50 = ccle_drugdata["IC50 (uM)"]
    ic50_normal = normalize(ic50)
    ccle_drugdata["ic50_normal"]=ic50_normal


    # label (sensitive or resistant)according to acta_normal
    ccle_drugdata = labeldata(ccle_drugdata,"acta_normal")


    # order ccle_drugdata by acta_normal
    ccle_drugdata=ccle_drugdata.sort_values(by="acta_normal", ascending=False)
    # ccle_drugdata["acta_normal"].plot(kind="bar")
    # plt.show()
    ccle_drugdata.rename(columns={"CCLE Cell Line Name":"CName"},inplace=True)
    # output
    # ccle_drugdata.to_csv('ccle_drugdata.csv')
    ccle_drugdata2 = ccle_drugdata[ccle_drugdata["label_y"]!=2]
    # ccle_drugdata2.to_csv('ccle_drugdata2.csv')
    drug_for1 = np.unique(ccle_drugdata[ccle_drugdata["label_y"]==1]["Compound"])
    drug_for0 = np.unique(ccle_drugdata[ccle_drugdata["label_y"]==0]["Compound"])
    drugs = [val for val in list(drug_for1) if val in list(drug_for0)]
    print(drugs)



    ########################################gene exprSet part#################################################
    # read gene exprset file
    L=[]
    file=open("gene.txt","r")
    # colname=file.readline()
    line=file.readline()
    while line:
        line=line.split()
        L.append(line)
        line=file.readline()
    # print(L)
    ccle_exprSet = pd.DataFrame(L)
    ccle_exprSet2=ccle_exprSet.T
    # print(ccle_exprSet2.iloc[0,1:18989])##row
    # print(ccle_exprSet2.iloc[2:1039,0])##column
    ccle_exprSet3 = ccle_exprSet2.drop([0,1]).reset_index(drop=True)
    ccle_exprSet3.columns=ccle_exprSet2.iloc[0,:].tolist()
    # ccle_exprSet3.to_csv('ccle_exprSet3.csv')
    # print(ccle_exprSet3.shape)
    # print(ccle_exprSet3)
    ccle_exprSet4 = pd.DataFrame(index=ccle_exprSet3.index)
    ccle_exprSet4[ccle_exprSet3.columns[0]]=ccle_exprSet3.iloc[:,0]
    for i in range(1,18989):
      ccle_exprSet4[ccle_exprSet3.columns[i]] =normalize01(ccle_exprSet3[ccle_exprSet3.columns[i]])
    # print(ccle_exprSet4)
    ccle_exprSet4.info()
    # ccle_exprSet4.to_csv('ccle_exprSet4.csv')









    ########################################copy number part#################################################
    # read copy number file
    ccle_cnaSet = pd.read_csv("cna.csv")
    ccle_cnaSet2 = ccle_cnaSet.drop(["Description"],axis=1)
    ccle_cnaSet3 = ccle_cnaSet2.T
    ccle_cnaSet4 = ccle_cnaSet3.drop(["Name"]).reset_index(drop=True)
    ccle_cnaSet4.columns=ccle_cnaSet3.iloc[0,:].tolist()
    ccle_cnaSet4.insert(0,"Name",ccle_cnaSet3.index[1:])
    # ccle_cnaSet4.to_csv("ccle_cnaSet4.csv")
    # print(ccle_cnaSet4)
    ccle_cnaSet4.info()

    ##################################generate x and y for each drug######################################
    drugs = ["AEW541","AZD0530","Erlotinib","LBW242","Lapatinib","Nilotinib","PD-0325901","PD-0332991","PF2341066","PLX4720","RAF265","Sorafenib","TAE684","TKI258","Topotecan"]
    # for i in cgp_drugdata["DRUG_ID"].tolist():
    #   if i not in drugs:
    #     drugs.append(i)
    # print(drugs)

    ####choose the same celllines in both gene and cna
    ccle_exprSet4.rename(columns={"Name":"CName"},inplace=True)##inplace=True表示在原数据上进行操作
    ccle_cnaSet4.rename(columns={"Name":"CName"},inplace=True)
    exprCName = ccle_exprSet4["CName"]
    # print(list(exprCName))
    # pd.DataFrame(list(exprCName)).to_csv("exprCName.csv")
    cnaCName = ccle_cnaSet4["CName"]
    # print(list(cnaCName))
    # pd.DataFrame(list(cnaCName)).to_csv("cnaCName.csv")
    conCName = [val for val in list(exprCName) if val in list(cnaCName)]
    conCName.remove("NCIH292_LUNG")
    conCName.remove("NCIH292_LUNG")
    print("con-cellline:",list(conCName))
    # pd.DataFrame(list(conCName)).to_csv("conCName.csv")



    #####for each drug
    all_drug_acc=[]
    for i in range(15):
        each_drug_acc = []
        drug=drugs[i]
        each_drug_acc.append(drug)

        each_drugdata = pd.DataFrame(ccle_drugdata2[ccle_drugdata2["Compound"]==drugs[i]])
        # each_drugdata.to_csv('each_drugdata.csv')
        # print(each_drugdata)


        ######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~prepare x and y(gene part)~~~~~~~~~~~~~~~~~~~~~~~~~####################
        ccle_exprSet5 = ccle_exprSet4[ccle_exprSet4['CName'].isin(list(conCName))]
        # ccle_exprSet5["CName"].to_csv("ccle_exprSet5.csv")
        each_data = pd.merge(each_drugdata,ccle_exprSet5,on="CName",how="inner")
        ###########################################test gcForest######################################33
        random.seed(5)
        randomCols = random.sample(range(16,19004),400)
        x = each_data.iloc[:,randomCols]
        y = each_data.iloc[:,15]
        x.index=each_data["CName"]
        y.index=each_data["CName"]
        x.sort_index()
        y.sort_index()
        folds_expr = five_fold(x.shape[0])
        # pd.DataFrame(folds_expr).to_csv("two/folds_expr.csv")



        ######~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~prepare x and y(cna part)~~~~~~~~~~~~~~~~~~~~~~~~~####################
        ccle_cnaSet5 = ccle_cnaSet4[ccle_cnaSet4['CName'].isin(list(conCName))]
        # ccle_cnaSet5["CName"].to_csv("ccle_cnaSet5.csv")
        each_cna_data = pd.merge(each_drugdata,ccle_cnaSet5,on="CName",how="inner")
        ###########################################test gcForest######################################33
        random.seed(5)
        randomCols = random.sample(range(16,46648),400)
        x_cna = each_cna_data.iloc[:,randomCols]
        y_cna = each_cna_data.iloc[:,15]
        x_cna.index=each_cna_data["CName"]
        y_cna.index=each_cna_data["CName"]
        x_cna.sort_index()
        y_cna.sort_index()
        folds_cna = five_fold(x_cna.shape[0])
        # pd.DataFrame(folds_cna).to_csv("two/folds_cna.csv")






        ############################################five fold##########################################
        feature_flag = ""
        five_results = []
        for j in range(5):
            each_fold_result=[]
            each_fold_result.append(j)
            X_test_expr = x.iloc[folds_expr[j],:]
            y_test_expr = y.iloc[folds_expr[j]]
            X_train_expr = x.iloc[list(set(range(x.shape[0])).difference(set(folds_expr[j]))),:]
            y_train_expr = y.iloc[list(set(range(x.shape[0])).difference(set(folds_expr[j])))]

            X_test_cna = x_cna.iloc[folds_cna[j],:]
            y_test_cna = y_cna.iloc[folds_cna[j]]
            X_train_cna = x_cna.iloc[list(set(range(x.shape[0])).difference(set(folds_cna[j]))),:]
            y_train_cna = y_cna.iloc[list(set(range(x.shape[0])).difference(set(folds_cna[j])))]

            X_train = np.hstack((X_train_expr,X_train_cna))
            y_train = y_train_expr
            X_test = np.hstack((X_test_expr,X_test_cna))
            y_test = y_test_expr
            # pd.DataFrame(X_train).to_csv("svm_rf/"+str(j)+str(drug)+"X_train.csv")
            # pd.DataFrame(y_train).to_csv("svm_rf/"+str(j)+str(drug)+"y_train.csv")
            pd.DataFrame(X_test).to_csv("svm_rf/"+str(j)+str(drug)+"X_test.csv")
            pd.DataFrame(y_test).to_csv("svm_rf/"+str(j)+str(drug)+"y_test.csv")


            ######################svm part################
            levels = np.unique(np.array(y_train))
            print("levels:",levels)
            File = open("svm_rf/"+str(j)+str(drug)+"svm.txt", "w")
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













