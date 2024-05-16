from __future__ import division
from sklearn import preprocessing
from numpy import *
import numpy as np
from scipy.spatial.distance import cdist


class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0

    def fit_transform(self, traindata):
        self._mean = traindata.mean(axis=0)
        self._std = traindata.std(axis=0)
        return (traindata - self._mean) / self._std

    def transform(self, testdata):
        return (testdata - self._mean) / self._std


class node_generator:
    def __init__(self, whiten=False):
        self.Wlist = []
        self.blist = []
        self.nonlinear = 0
        self.whiten = whiten

    def sigmoid(self, data):
        return 1.0 / (1 + np.exp(-data))

    def linear(self, data):
        return data

    def tanh(self, data):
        return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

    def relu(self, data):
        return np.maximum(data, 0)

    def orth(self, W):

        for i in range(0, W.shape[1]):
            w = np.mat(W[:, i].copy()).T
            w_sum = 0
            for j in range(i):
                wj = np.mat(W[:, j].copy()).T

                w_sum += (w.T.dot(wj))[0, 0] * wj  # [0,0]就是求矩阵相乘的一元数
            w -= w_sum
            w = w / np.sqrt(w.T.dot(w))
            W[:, i] = np.ravel(w)

        return W

    def generator(self, shape, times):
        for i in range(times):
            W = 2 * random.random(size=shape) - 1
            if self.whiten == True:
                W = self.orth(W)
            b = 2 * random.random() - 1
            yield (W, b)

    def generator_nodes(self, data, times, batchsize, nonlinear):  # 将特征结点和增强结点构建起来
        self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]

        self.nonlinear = {'linear': self.linear,
                          'sigmoid': self.sigmoid,
                          'tanh': self.tanh,
                          'relu': self.relu
                          }[nonlinear]
        nodes = self.nonlinear(data.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            nodes = np.column_stack((nodes, self.nonlinear(data.dot(self.Wlist[i]) + self.blist[i])))
        return nodes

    def transform(self, testdata):
        testnodes = self.nonlinear(testdata.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i]) + self.blist[i])))
        return testnodes

    def update(self, otherW, otherb):
        self.Wlist += otherW
        self.blist += otherb


class MPOS_RVFL:
    def __init__(self, Ne, N2, enhence_function, reg, gamma, n_anchor, sigma):
        self.nHiddenNeurons = int(Ne * N2)
        self._Ne = Ne
        self._N2 = N2
        self._reg = reg
        self._gamma = gamma
        self.anchor = n_anchor
        self._sigma = sigma
        self._enhence_function = enhence_function

        self.p_global = []
        self.c_weight = []

        self.Xf = []
        self.Af = []
        self.Tf = []
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.enhence_generator = node_generator(whiten=False)

    def cal_S(self, X, K):
        """

        :param X: original data
        :param K: anchors data
        :return: normalized similarity matrix
        """
        sigma = self._sigma
        k = len(K)

        n, _ = X.shape
        d, _ = K.shape

        # Calculate the Euclidean distance matrix
        distance_matrix = cdist(X, K, metric='euclidean')

        # Calculate s_ij
        s_matrix = np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                if j in np.argsort(distance_matrix[i])[:k]:
                    s_matrix[i, j] = np.exp(-np.linalg.norm(X[i] - K[j]) ** 2 / (2 * sigma ** 2))

        row_sums = s_matrix.sum(axis=1) + 1e-10
        s_matrix_normalized = s_matrix / row_sums[:, np.newaxis]

        return s_matrix_normalized


    def softmax_norm(self, array):
        exp_array = np.exp(array)
        sum_exp_array = np.sum(exp_array, axis=1, keepdims=True)
        softmax_array = exp_array / sum_exp_array
        return softmax_array

    def decode(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)

    def transform(self, data):
        enhencedata = self.enhence_generator.transform(data)
        inputdata = np.column_stack((enhencedata, data))
        return inputdata

    def cal_weight(self):
        # Acquisition of minority weighting factors
        types, counts = np.unique(self.p_global, return_counts=True)
        num_types = len(types)
        major_type_counts_1 = np.max(counts)
        w = np.ones(num_types)
        for i in range(num_types):
            w[i] = major_type_counts_1/counts[i]
        self.c_weight = w

    def fit(self, X, Y):
        # Data normalization and random mapping
        X_enc = self.normalscaler.fit_transform(X)
        Y_enc = self.onehotencoder.fit_transform(np.mat(Y).T)
        H = self.enhence_generator.generator_nodes(X_enc, self._Ne, self._N2, self._enhence_function)
        self.A = np.column_stack((H, X_enc))

        # Minority Anchors Prioritization
        Y_types, count_types = np.unique(Y, return_counts=True)
        count_min = np.min(count_types)
        # If data imbalance is not severe, randomly choose samples from each types
        if count_min >= int(self.anchor/len(Y_types)):
            select_index = []
            for p in Y_types:
                select_index.extend(np.random.choice(np.where(Y == p)[0], int(self.anchor/len(Y_types)), replace=False))
        # If data imbalance is severe, priority will be given to minority class
        else:
            select_index = []
            for p in Y_types:
                if p != 0:
                    select_index.extend(np.random.choice(np.where(Y == p)[0], count_min, replace=False))
            remain_num = self.anchor - len(select_index)
            select_index.extend(np.random.choice(np.where(Y == 0)[0], remain_num, replace=False))

        self.Tf = Y_enc[select_index]
        self.Af = self.A[select_index]
        self.Xf = X[select_index]

        r, w = self.A.T.dot(self.A).shape
        self.S = self.cal_S(X, self.Xf)
        self.C = np.eye(X.shape[0])

        # Count the number of sample labels
        self.p_global = Y.tolist()

        # Set weights for C matrix
        self.cal_weight()
        for t in range(len(self.C)):
            current_label = int(Y[t])
            current_weight = self.c_weight[current_label]
            self.C[t][t] = current_weight

        # Calculation of intermediate variables
        self.G = self.A.T.dot(self.C).dot(self.A) + self._reg * np.eye(r) + self._gamma * (
                    self.A - self.S.dot(self.Af)).T.dot(self.A - self.S.dot(self.Af))
        self.Omega = self.A.T.dot(self.C).dot(Y_enc)
        self.G_inv = np.linalg.inv(self.G)
        self.W = self.G_inv.dot(self.Omega)

    def MP_index(self, p_label, p_conf_level, C):
        for index in range(len(p_label)):
            current_p_label = int(p_label[index])
            current_p_conf_level = p_conf_level[index]
            if current_p_label == 0:
                if current_p_conf_level >= 0.98:
                    C[index][index] = current_p_conf_level
            else:
                if current_p_conf_level >= 0.90:
                    current_p_weight = self.c_weight[current_p_label]
                    C[index][index] = current_p_conf_level * current_p_weight
        return C

    def Pseudo_label(self, S_on):
        """

        :param S_on: normalized similarity matrix at time t
        :return: pseudo-label vector and corresponding confidence level matrix
        """
        column_sums = np.sum(S_on, axis=0)
        column_sums[column_sums == 0] = 1

        P_on = S_on / column_sums
        f_U = P_on.T @ self.Tf
        p_label = np.argmax(f_U, axis=1)
        p_conf_level = np.max(f_U, axis=1)
        return p_label, p_conf_level

    def partial_fit(self, X_at, Y_at=None):
        X_at_enc = self.normalscaler.transform(X_at)
        A_at = self.transform(X_at_enc)
        S_at = self.cal_S(X_at, self.Xf)
        C_at = np.zeros((len(X_at), len(X_at)))

        # Calculate pseudo-label
        pseudo_label, pseudo_conf_level = self.Pseudo_label(S_at.T)
        self.p_global.extend(pseudo_label.tolist())
        self.cal_weight()
        T_at = self.onehotencoder.transform(np.mat(pseudo_label).T)
        # Assign pseudo-labels that meet threshold conditions
        C_at = self.MP_index(pseudo_label, pseudo_conf_level, C_at)

        # incremental update
        U_ts = A_at.T.dot(C_at)
        V_ts = A_at
        U_tu = self._gamma * (A_at - S_at.dot(self.Af)).T
        V_tu = A_at - S_at.dot(self.Af)

        Psi_inv = self.G_inv - (self.G_inv.dot(U_ts)).dot(
            np.linalg.inv(np.eye(V_ts.shape[0]) + V_ts.dot(self.G_inv).dot(U_ts))).dot(V_ts).dot(self.G_inv)
        self.G_inv = Psi_inv - (Psi_inv.dot(U_tu)).dot(
            np.linalg.inv(np.eye(V_tu.shape[0]) + V_tu.dot(Psi_inv).dot(U_tu))).dot(V_tu).dot(Psi_inv)
        self.Omega = self.Omega + A_at.T.dot(C_at).dot(T_at)
        self.W = self.G_inv.dot(self.Omega)

    def predict(self, testdata):
        logit = self.predict_proba(testdata)
        return self.decode(self.softmax_norm(logit))

    def predict_proba(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_inputdata = self.transform(testdata)
        org_prediction = test_inputdata.dot(self.W)
        return self.softmax_norm(org_prediction)
