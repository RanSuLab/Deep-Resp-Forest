#!usr/bin/env python
"""
Version : 0.1.6
Date : 15th April 2017

Author : Pierre-Yves Lablanche
Email : plablanche@aims.ac.za
Affiliation : African Institute for Mathematical Sciences - South Africa
              Stellenbosch University - South Africa

License : MIT

Status : Not Under Active Development

Description :
Python3 implementation of the gcForest algorithm preesented in Zhou and Feng 2017
(paper can be found here : https://arxiv.org/abs/1702.08835 ).
It uses the typical scikit-learn syntax  with a .fit() function for training
and a .predict() function for predictions.

"""
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
##importances
import pandas as pd
__author__ = "Pierre-Yves Lablanche"
__email__ = "plablanche@aims.ac.za"
__license__ = "MIT"
__version__ = "0.1.6"
#__status__ = "Development"


# noinspection PyUnboundLocalVariable
class gcForest(object):
    # 初始化各种参数
    def __init__(self, shape_1X=None, n_mgsRFtree=60, window=None, stride=1,
                 cascade_test_size=0.2, n_cascadeRF=2, n_cascadeRFtree=100, cascade_layer=np.inf,
                 min_samples_mgs=0.1, min_samples_cascade=0.05, tolerance=0.0, n_jobs=1,levels=[0,1,2],f=None):
        """ gcForest Classifier.

        :param shape_1X: int or tuple list or np.array (default=None)
            Shape of a single sample element [n_lines, n_cols]. Required when calling mg_scanning!
            For sequence data a single int can be given.

        :param n_mgsRFtree: int (default=30)
            Number of trees in a Random Forest during Multi Grain Scanning.

        :param window: int (default=None)
            List of window sizes to use during Multi Grain Scanning.
            If 'None' no slicing will be done.

        :param stride: int (default=1)
            Step used when slicing the data.

        :param cascade_test_size: float or int (default=0.2)
            Split fraction or absolute number for cascade training set splitting.

        :param n_cascadeRF: int (default=2)
            Number of Random Forests in a cascade layer.
            For each pseudo Random Forest a complete Random Forest is created, hence
            the total number of Random Forests in a layer will be 2*n_cascadeRF.

        :param n_cascadeRFtree: int (default=101)
            Number of trees in a single Random Forest in a cascade layer.

        :param min_samples_mgs: float or int (default=0.1)
            Minimum number of samples in a node to perform a split
            during the training of Multi-Grain Scanning Random Forest.
            If int number_of_samples = int.
            If float, min_samples represents the fraction of the initial n_samples to consider.

        :param min_samples_cascade: float or int (default=0.1)
            Minimum number of samples in a node to perform a split
            during the training of Cascade Random Forest.
            If int number_of_samples = int.
            If float, min_samples represents the fraction of the initial n_samples to consider.

        :param cascade_layer: int (default=np.inf)
            mMximum number of cascade layers allowed.
            Useful to limit the contruction of the cascade.

        :param tolerance: float (default=0.0)
            Accuracy tolerance for the casacade growth.
            If the improvement in accuracy is not better than the tolerance the construction is
            stopped.

        :param n_jobs: int (default=1)
            The number of jobs to run in parallel for any Random Forest fit and predict.
            If -1, then the number of jobs is set to the number of cores.

        :param levels: list (default=[0,1,2])****new****
            The levels of labels.
        :param f:File(None)****new****
            The output file.
        """
        setattr(self, 'shape_1X', shape_1X)
        setattr(self, 'n_layer', 0)
        setattr(self, '_n_samples', 0)
        setattr(self, 'n_cascadeRF', int(n_cascadeRF))
        if isinstance(window, int):
            setattr(self, 'window', [window])
        elif isinstance(window, list):
            setattr(self, 'window', window)
        setattr(self, 'stride', stride)
        setattr(self, 'cascade_test_size', cascade_test_size)
        setattr(self, 'n_mgsRFtree', int(n_mgsRFtree))
        setattr(self, 'n_cascadeRFtree', int(n_cascadeRFtree))
        setattr(self, 'cascade_layer', cascade_layer)
        setattr(self, 'min_samples_mgs', min_samples_mgs)
        setattr(self, 'min_samples_cascade', min_samples_cascade)
        setattr(self, 'tolerance', tolerance)
        setattr(self, 'n_jobs', n_jobs)
        setattr(self, 'levels', levels)
        setattr(self, 'f', f)

    # 训练模型
    def fit(self, X, y):
        """ Training the gcForest on input data X and associated target y.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array
            1D array containing the target values.
            Must be of shape [n_samples]
        """
        #样本量和标签数相同
        if np.shape(X)[0] != len(y):
            raise ValueError('Sizes of y and X do not match.')
        #切片+预测
        mgs_X = self.mg_scanning(X, y)
        #层级森林
        _ = self.cascade_forest(mgs_X, y)

    def predict_proba(self, X):
        """ Predict the class probabilities of unknown samples X.

        :param X: np.array
            Array containing the input samples.
            Must be of the same shape [n_samples, data] as the training inputs.

        :return: np.array
            1D array containing the predicted class probabilities for each input sample.
        """
        mgs_X = self.mg_scanning(X)
        cascade_all_pred_prob = self.cascade_forest(mgs_X)
        predict_proba = np.mean(cascade_all_pred_prob, axis=0)

        return predict_proba

    def predict(self, X):
        """ Predict the class of unknown samples X.

        :param X: np.array
            Array containing the input samples.
            Must be of the same shape [n_samples, data] as the training inputs.

        :return: np.array
            1D array containing the predicted class for each input sample.
        """
        pred_proba = self.predict_proba(X=X)
        predictions = self.levels[np.argmax(pred_proba, axis=1)]
        return predictions
    #多粒度扫描  输入数据
    def mg_scanning(self, X, y=None):
        """ Performs a Multi Grain Scanning on input data.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)

        :return: np.array
            Array of shape [n_samples, .. ] containing Multi Grain Scanning sliced data.
        """
        # 样本量
        setattr(self, '_n_samples', np.shape(X)[0])
        #每个样本的格式
        shape_1X = getattr(self, 'shape_1X')
        #判断是否为数列
        if isinstance(shape_1X, int):
            shape_1X = [1,shape_1X]
        #设置切片窗口
        if not getattr(self, 'window'):
            setattr(self, 'window', [shape_1X[1]])

        mgs_pred_prob = []
        # 在1:window里面循环切片
        for wdw_size in getattr(self, 'window'):
            #切好的每一片+预测
            print("wdw_size:",wdw_size)
            wdw_pred_prob = self.window_slicing_pred_prob(X, wdw_size, shape_1X, y=y)
            # print("wdw_pred_prob:",wdw_pred_prob)
            #添加
            mgs_pred_prob.append(wdw_pred_prob)
        # print("mgs_pred_prob:")
        return mgs_pred_prob

    def window_slicing_pred_prob(self, X, window, shape_1X, y=None):
        """ Performs a window slicing of the input data and send them through Random Forests.
        If target values 'y' are provided sliced data are then used to train the Random Forests.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample.

        :param y: np.array (default=None)
            Target values. If 'None' no training is done.

        :return: np.array
            Array of size [n_samples, ..] containing the Random Forest.
            prediction probability for each input sample.
        """
        n_tree = getattr(self, 'n_mgsRFtree')
        min_samples = getattr(self, 'min_samples_mgs')
        stride = getattr(self, 'stride')
        # 切图片
        if shape_1X[0] > 1:
            print('Slicing Images...')
            sliced_X, sliced_y = self._window_slicing_img(X, window, shape_1X, y=y, stride=stride)
        # 切序列
        else:
            print('Slicing Sequence...')
            sliced_X, sliced_y = self._window_slicing_sequence(X, window, shape_1X, y=y, stride=stride)

        if y is not None:
            n_jobs = getattr(self, 'n_jobs')
            prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                         min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs,random_state=5)
            crf = RandomForestClassifier(n_estimators=n_tree, max_features=1,
                                         min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs,random_state=5)
            print('Training MGS Random Forests...')
            prf.fit(sliced_X, sliced_y)
            crf.fit(sliced_X, sliced_y)
            setattr(self, '_mgsprf_{}'.format(window), prf)
            setattr(self, '_mgscrf_{}'.format(window), crf)
            pred_prob_prf = prf.oob_decision_function_
            pred_prob_crf = crf.oob_decision_function_



        if hasattr(self, '_mgsprf_{}'.format(window)) and y is None:
            prf = getattr(self, '_mgsprf_{}'.format(window))
            crf = getattr(self, '_mgscrf_{}'.format(window))
            pred_prob_prf = prf.predict_proba(sliced_X)
            pred_prob_crf = crf.predict_proba(sliced_X)



        pred_prob = np.c_[pred_prob_prf, pred_prob_crf]
        # print("pred_prob:",pred_prob)
        return pred_prob.reshape([getattr(self, '_n_samples'), -1])

    def _window_slicing_img(self, X, window, shape_1X, y=None, stride=1):
        """ Slicing procedure for images

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample [n_lines, n_cols].

        :param y: np.array (default=None)
            Target values.

        :param stride: int (default=1)
            Step used when slicing the data.

        :return: np.array and np.array
            Arrays containing the sliced images and target values (empty if 'y' is None).
        """
        if any(s < window for s in shape_1X):
            raise ValueError('window must be smaller than both dimensions for an image')

        len_iter_x = np.floor_divide((shape_1X[1] - window), stride) + 1
        len_iter_y = np.floor_divide((shape_1X[0] - window), stride) + 1
        iterx_array = np.arange(0, stride*len_iter_x, stride)
        itery_array = np.arange(0, stride*len_iter_y, stride)

        ref_row = np.arange(0, window)
        ref_ind = np.ravel([ref_row + shape_1X[1] * i for i in range(window)])
        inds_to_take = [ref_ind + ix + shape_1X[1] * iy
                        for ix, iy in itertools.product(iterx_array, itery_array)]

        sliced_imgs = np.take(X, inds_to_take, axis=1).reshape(-1, window**2)

        if y is not None:
            sliced_target = np.repeat(y, len_iter_x * len_iter_y)
        elif y is None:
            sliced_target = None

        return sliced_imgs, sliced_target

    def _window_slicing_sequence(self, X, window, shape_1X, y=None, stride=1):
        """ Slicing procedure for sequences (aka shape_1X = [.., 1]).

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param window: int
            Size of the window to use for slicing.

        :param shape_1X: list or np.array
            Shape of a single sample [n_lines, n_col].

        :param y: np.array (default=None)
            Target values.

        :param stride: int (default=1)
            Step used when slicing the data.

        :return: np.array and np.array
            Arrays containing the sliced sequences and target values (empty if 'y' is None).
        """
        # 序列比窗口长度小
        if shape_1X[1] < window:
            raise ValueError('window must be smaller than the sequence dimension')
        #切成多少个片段
        len_iter = np.floor_divide((shape_1X[1] - window), stride) + 1
        iter_array = np.arange(0, stride*len_iter, stride)

        ind_1X = np.arange(np.prod(shape_1X))
        inds_to_take = [ind_1X[i:i+window] for i in iter_array]
        sliced_sqce = np.take(X, inds_to_take, axis=1).reshape(-1, window)

        if y is not None:
            sliced_target = np.repeat(y, len_iter)
        elif y is None:
            sliced_target = None

        return sliced_sqce, sliced_target
        # 切好的X和y

    def cascade_forest(self, X1,X2,X3,X4, y=None):
        """ Perform (or train if 'y' is not None) a cascade forest estimator.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)
            Target values. If 'None' perform training.

        :return: np.array
            1D array containing the predicted class for each input sample.
        """
        if y is not None:
            setattr(self, 'n_layer', 0)
            test_size = getattr(self, 'cascade_test_size')
            max_layers = getattr(self, 'cascade_layer')
            tol = getattr(self, 'tolerance')

            X_train1, X_test1, y_train, y_test = train_test_split(X1, y, test_size=test_size,random_state=5)
            print("X_train1:",X_train1)
            print("y_train:",y_train)
            X_train2, X_test2, y_train, y_test = train_test_split(X2, y, test_size=test_size,random_state=5)
            print("X_train2:",X_train2)
            print("y_train:",y_train)
            X_train3, X_test3, y_train, y_test = train_test_split(X3, y, test_size=test_size,random_state=5)
            print("X_train3:",X_train3)
            print("y_train:",y_train)
            X_train4, X_test4, y_train, y_test = train_test_split(X4, y, test_size=test_size,random_state=5)
            print("X_train4:",X_train4)
            print("y_train:",y_train)
            self.n_layer += 1
            if self.n_layer%4==1:
              prf_crf_pred_ref = self._cascade_layer(X_train1, y_train)
              accuracy_ref = self._cascade_evaluation(X_test1,X_test2,X_test3,X_test4, y_test)
              feat_arr = self._create_feat_arr(X_train2, prf_crf_pred_ref)
            self.n_layer += 1
            if self.n_layer%4==2:
              prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)
              accuracy_layer = self._cascade_evaluation(X_test1,X_test2,X_test3,X_test4, y_test)

            while accuracy_layer > (accuracy_ref + tol) and self.n_layer <= max_layers:
                accuracy_ref = accuracy_layer
                prf_crf_pred_ref = prf_crf_pred_layer
                if self.n_layer%4==1:
                    feat_arr = self._create_feat_arr(X_train2, prf_crf_pred_ref)
                elif self.n_layer%4==2:
                    feat_arr = self._create_feat_arr(X_train3, prf_crf_pred_ref)
                elif self.n_layer%4==3:
                    feat_arr = self._create_feat_arr(X_train4, prf_crf_pred_ref)
                elif self.n_layer%4==0:
                    feat_arr = self._create_feat_arr(X_train1, prf_crf_pred_ref)
                print(self.n_layer,feat_arr)
                self.n_layer += 1
                prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)
                # print("prf_crf_pred_layer:",prf_crf_pred_layer)
                accuracy_layer = self._cascade_evaluation(X_test1,X_test2,X_test3,X_test4, y_test)
                # print("accuracy_layer:",accuracy_layer)

            if accuracy_layer < accuracy_ref :
                n_cascadeRF = getattr(self, 'n_cascadeRF')
                for irf in range(n_cascadeRF):
                    delattr(self, '_casprf{}_{}'.format(self.n_layer, irf))
                    delattr(self, '_cascrf{}_{}'.format(self.n_layer, irf))
                self.n_layer -= 1
            # print("prf_crf_pred_ref:",prf_crf_pred_ref)
        elif y is None:
            at_layer = 1
            if at_layer%4==1:
                prf_crf_pred_ref = self._cascade_layer(X1, layer=at_layer)
            elif at_layer%4==2:
                prf_crf_pred_ref = self._cascade_layer(X2, layer=at_layer)
            elif at_layer%4==3:
                prf_crf_pred_ref = self._cascade_layer(X3, layer=at_layer)
            elif at_layer%4==0:
                prf_crf_pred_ref = self._cascade_layer(X4, layer=at_layer)
            while at_layer < getattr(self, 'n_layer'):
                if at_layer%4==1:
                    feat_arr = self._create_feat_arr(X2, prf_crf_pred_ref)
                elif at_layer%4==2:
                    feat_arr = self._create_feat_arr(X3, prf_crf_pred_ref)
                elif at_layer%4==3:
                    feat_arr = self._create_feat_arr(X4, prf_crf_pred_ref)
                elif at_layer%4==0:
                    feat_arr = self._create_feat_arr(X1, prf_crf_pred_ref)
                at_layer += 1
                prf_crf_pred_ref = self._cascade_layer(feat_arr, layer=at_layer)

        return prf_crf_pred_ref

    def _cascade_layer(self, X, y=None, layer=0):
        """ Cascade layer containing Random Forest estimators.
        If y is not None the layer is trained.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param y: np.array (default=None)
            Target values. If 'None' perform training.

        :param layer: int (default=0)
            Layer indice. Used to call the previously trained layer.

        :return: list
            List containing the prediction probabilities for all samples.
        """
        n_tree = getattr(self, 'n_cascadeRFtree')
        n_cascadeRF = getattr(self, 'n_cascadeRF')
        min_samples = getattr(self, 'min_samples_cascade')

        n_jobs = getattr(self, 'n_jobs')
        prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                     min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs,random_state=5)
        crf = RandomForestClassifier(n_estimators=n_tree, max_features=1,
                                     min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs,random_state=5)

        prf_crf_pred = []
        if y is not None:
            print('Adding/Training Layer, n_layer={}'.format(self.n_layer))
            for irf in range(n_cascadeRF):
                prf.fit(X, y)
                crf.fit(X, y)
                setattr(self, '_casprf{}_{}'.format(self.n_layer, irf), prf)
                setattr(self, '_cascrf{}_{}'.format(self.n_layer, irf), crf)
                prf_crf_pred.append(prf.oob_decision_function_)
                prf_crf_pred.append(crf.oob_decision_function_)
        elif y is None:
            for irf in range(n_cascadeRF):
                prf = getattr(self, '_casprf{}_{}'.format(layer, irf))
                crf = getattr(self, '_cascrf{}_{}'.format(layer, irf))
                prf_crf_pred.append(prf.predict_proba(X))
                prf_crf_pred.append(crf.predict_proba(X))
        # print("_cascade_layer prf_crf_pred:",prf_crf_pred)
        return prf_crf_pred

    def _cascade_evaluation(self, X_test1,X_test2,X_test3,X_test4, y_test):
        """ Evaluate the accuracy of the cascade using X and y.

        :param X_test: np.array
            Array containing the test input samples.
            Must be of the same shape as training data.

        :param y_test: np.array
            Test target values.

        :return: float
            the cascade accuracy.
        """
        casc_pred_prob = np.mean(self.cascade_forest(X_test1,X_test2,X_test3,X_test4), axis=0)
        casc_pred = self.levels[np.argmax(casc_pred_prob, axis=1)]
        casc_accuracy = accuracy_score(y_true=y_test, y_pred=casc_pred)
        print('Layer validation accuracy = {}'.format(casc_accuracy))
        self.f.write('Layer validation accuracy = {}'.format(casc_accuracy) + "\n")
        return casc_accuracy

    def _create_feat_arr(self, X, prf_crf_pred):
        """ Concatenate the original feature vector with the predicition probabilities
        of a cascade layer.

        :param X: np.array
            Array containing the input samples.
            Must be of shape [n_samples, data] where data is a 1D array.

        :param prf_crf_pred: list
            Prediction probabilities by a cascade layer for X.

        :return: np.array
            Concatenation of X and the predicted probabilities.
            To be used for the next layer in a cascade forest.
        """
        swap_pred = np.swapaxes(prf_crf_pred, 0, 1)
        add_feat = swap_pred.reshape([np.shape(X)[0], -1])
        feat_arr = np.concatenate([add_feat, X], axis=1)
        # print("feat_arr:",feat_arr)

        return feat_arr
