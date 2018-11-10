
# -*- coding: utf-8 -*-

import itertools

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt



import sklearn.metrics as metrics

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split



from GCForest_two import gcForest
import random
from sklearn.metrics import accuracy_score



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
    # drugs = []
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
    for i in range(20):
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
        pd.DataFrame(folds_expr).to_csv("two1/folds_expr.csv")



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
        pd.DataFrame(folds_cna).to_csv("two1/folds_cna.csv")






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
            X_test.to_csv("two1/"+str(j)+str(drug)+"_X_test_expr.csv")
            y_test.to_csv("two1/"+str(j)+str(drug)+"_y_test_expr.csv")
            train_cellline = X_train.index
            print("train_cellline:",train_cellline)
            test_cellline = X_test.index
            print("test_cellline:",test_cellline)

            X_test_cna = x_cna.iloc[folds_cna[j],:]
            y_test_cna = y_cna.iloc[folds_cna[j]]
            X_train_cna = x_cna.iloc[list(set(range(x.shape[0])).difference(set(folds_cna[j]))),:]
            y_train_cna = y_cna.iloc[list(set(range(x.shape[0])).difference(set(folds_cna[j])))]
            X_test_cna.to_csv("two1/"+str(j)+str(drug)+"_X_test_cna.csv")
            y_test_cna.to_csv("two1/"+str(j)+str(drug)+"_y_test_cna.csv")


            ######################mgs expr part################
            levels = np.unique(np.array(y_train))
            print("levels:",levels)
            File = open("two1/"+str(j)+str(drug)+".txt", "w")
            File.write("levels:"+str(levels)+"\n")
            clf = gcForest(shape_1X=(1, 400),window=[100],stride=2,levels=levels,f=File)
            if np.shape(X_train)[0] != len(y_train):
                raise ValueError('Sizes of y and X do not match.')
            expr_mgs_X = clf.mg_scanning(np.array(X_train), np.array(y_train))
            print(expr_mgs_X)
            expr_window1 = expr_mgs_X[0]
            print("expr_window1：",expr_window1)
            expr_mgs_X_test = clf.mg_scanning(np.array(X_test))
            expr_window1_test = expr_mgs_X_test[0]






            ######################mgs cna part################
            clf = gcForest(shape_1X=(1, 400),window=[100],stride=2,levels=levels,f=File)
            if np.shape(X_train_cna)[0] != len(y_train_cna):
                raise ValueError('Sizes of y and X do not match.')
            cna_mgs_X = clf.mg_scanning(np.array(X_train_cna), np.array(y_train_cna))
            print(cna_mgs_X)
            cna_window1 = cna_mgs_X[0]
            print("cna_window1：",cna_window1)
            cna_mgs_X_test = clf.mg_scanning(np.array(X_test_cna))
            cna_window1_test = cna_mgs_X_test[0]



            ######################cascade expr_cna part################
            train_predict_y = clf.cascade_forest(expr_window1,cna_window1,np.array(y_train_cna))
            if train_predict_y=="no_features":
                feature_flag = "no_features"
                print("drug"+str(drug)+":all feature importances are zeros")
                break



            #####################predict values###########################
            cascade_all_pred_prob = clf.cascade_forest(expr_window1_test,cna_window1_test)
            predict_proba = np.mean(cascade_all_pred_prob, axis=0)
            pd.DataFrame(predict_proba).to_csv("two1/"+str(j)+str(drug)+"_predict_proba.csv")
            predictions = levels[np.argmax(predict_proba, axis=1)]
            pd.DataFrame(predictions).to_csv("two1/"+str(j)+str(drug)+"_predictions.csv")
            prediction_accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
            each_fold_result.append(prediction_accuracy)
            print('Layer validation accuracy = {}'.format(prediction_accuracy))
            File.write('prediction_accuracy = {}'.format(prediction_accuracy)+"\n")
            # tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
            # specificity = tn /float(tn+fp)
            # each_fold_result.append(specificity)
            # print("specificity:",specificity)
            # File.write('specificity = {}'.format(specificity)+"\n")
            # sensitivity= tp/float(tp+fn)
            # each_fold_result.append(sensitivity)
            # print("sensitivity:",sensitivity)
            # File.write('sensitivity = {}'.format(sensitivity)+"\n")
            File.close()
            five_results.append(each_fold_result)
        print(feature_flag)
        print(feature_flag is "no_features")
        if feature_flag is "no_features":
            continue
        pd.DataFrame(five_results).to_csv("two1/"+str(drug)+"five_results.csv")
        print(five_results)
        each_drug_acc.append(pd.DataFrame(five_results).iloc[:,1].mean())
        all_drug_acc.append(each_drug_acc)
    pd.DataFrame(all_drug_acc).to_csv("two1/all_drug_acc.csv")













