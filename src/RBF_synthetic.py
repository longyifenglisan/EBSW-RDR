import time
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.uniform import Uniform
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from src.classifiers import KRR
from src.training import train_multiple_kernels
from src.EBSW_kernels import k_sw_rf
from src.MMDkernels import k_MMD
from sklearn.model_selection import GridSearchCV
from src.RBF_kernels import K_RBF
import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def GMM(means, covs, p):
    """GMM in arbitrary dimension d with K components.

    Args:
        means (Tensor): shape (K,d)
        covs (Tensor): (K,d,d), var-cov matrices
        p (Tensor): shape (K,), latent probabilities (or relative weights)

    Returns:
        MixtureSameFamily
    """
    assert p.shape[0] == means.shape[0]
    assert p.shape[0] == covs.shape[0]
    assert means.shape[1] == covs.shape[1]
    assert means.shape[1] == covs.shape[2]
    mix = Categorical(p)
    comp = MultivariateNormal(means, covs)
    gmm = MixtureSameFamily(mix, comp)
    return gmm


def sample_moments(T, d, K_max):
    """Sample moments of T GMMs in dimension d.
    Args:
        T (int): number of tasks
        d (int): input dimension
        K_max (int): maximum number of components
    Returns:
        y (T,), means (list), covs (list)
    """
    means_l = []
    covs_l = []
    y = Categorical(torch.ones(K_max)).sample((T,))  # uniform draw of modes
    for t in range(T):
        K = y[t] + 1
        means = Uniform(low=-5., high=5.).sample((K, d))
        covs = Uniform(low=-1., high=1.).sample((K, d, d))
        a = Uniform(low=1., high=4.).sample((K,))
        B = Uniform(low=0., high=1.).sample((K, d))
        covs = a.view(-1, 1, 1) * covs @ torch.transpose(covs, 1, 2) + torch.stack([torch.diag(b) for b in B])
        means_l.append(means)
        covs_l.append(covs)
    y = F.one_hot(y.to(torch.int64))

    return means_l, covs_l, y


def sample_from_moments(means, covs, n, K_max, epoch):
    T = len(means)
    d = means[0].shape[-1]
    X = torch.zeros((T, epoch, n, d))
    for t in range(T):
        K = means[t].shape[0]
        for e in range(epoch):
            X[t, e] = GMM(means[t], covs[t], torch.ones(K)).sample((n,))
    return X


def EBSW_training(X, y, X_val, y_val, X_test, y_test, p, epoch, v):
    hp_kernel = torch.logspace(-5, 4, 14)  # gamma的取值空间，带宽！！！
    p_lambda = 25  ###
    lambda_params = torch.logspace(-8, 2, p_lambda)  # lambd的取值空间，正则化强度！！！
    hp_params = lambda_params
    rmse_l = []
    d = X[0].shape[-1]

    for e in range(epoch):
        if v: print("EBSW epoch {}/{}".format(e + 1, epoch))
        k_class = k_sw_rf(100,
                          100,
                          non_uniform=False,
                          d_in=d,
                          p=p,
                          true_rf=True)
        assert k_class.r > 0

        K_train, K_val = k_class.get_grams(X[:, e, :, :],
                                           X_val[:, e, :, :],
                                           hp_kernel)

        results, hp_clf_opt, hps, w_opt = train_multiple_kernels(
            K_train,
            K_val,
            y,
            y_val,
            hp_params,
            hp_kernel,
            subsample=1)
        gammas = torch.tensor(hps["KRR"]["hp_kernel"])
        # print("Gammas rbf", gammas)
        start_time = time.time()
        K_test = k_class.get_cross_gram(X[:, e, :, :], X_test[:, e, :, :], gammas)
        if v: print("time elapsed computing both Gram matrices {:.2f}s (shape: {})\n".format(time.time() - start_time,
                                                                                             K_test.shape))
        clf = KRR()
        clf.fitted = True
        clf.alpha_ = w_opt
        y_test_pred_idx = clf.predict(K_test[0])
        acc = accuracy_score(y_test_pred_idx, y_test.argmax(dim=1))
        rmse = np.sqrt(torch.sum((y_test_pred_idx - y_test.argmax(dim=1)) ** 2) / y_test.shape[0])
        rmse_l.append(rmse.item())
        mean_rmse = sum(rmse_l) / len(rmse_l)
    return mean_rmse, acc.item(), y_test_pred_idx


def mmd_training(X, y, X_val, y_val, X_test, y_test, epoch, v):
    hp_in = torch.logspace(-6, 2, 14)
    hp_out = torch.logspace(-3, 2, 7)
    hp_kernel = torch.cartesian_prod(hp_out, hp_in)
    p_lambda = 25
    hp_params = torch.logspace(-8, 2, p_lambda)
    # hp_params = hp_params
    rmse_l = []

    for e in range(epoch):
        if v: print("MMD epoch {}/{}".format(e + 1, epoch))
        k_class = k_MMD(hp_in, hp_out, non_uniform=False)
        K_train, K_val = k_class.get_grams_gauss(X[:, e, :, :], X_val[:, e, :, :])
        K_train = K_train.flatten(0, 1)
        K_val = K_val.flatten(0, 1)
        _, _, hps, w_opt = train_multiple_kernels(K_train,
                                                  K_val,
                                                  y,
                                                  y_val,
                                                  hp_params,
                                                  hp_kernel,
                                                  subsample=1)

        hp = torch.stack(hps["KRR"]["hp_kernel"])
        hp_out = hp[:, 0]
        hp_in = hp[:, 1]
        # print("H", hp_in, hp_out)
        k_class = k_MMD(hp_in, hp_out, non_uniform=False)
        start_time = time.time()
        K_test = k_class.get_cross_gram_gauss(X[:, e, :, :], X_test[:, e, :, :])
        K_test = torch.stack([K_test[i, i] for i in range(len(K_test))])
        # print(K_test.shape)
        if v: print("time elapsed computing both Gram matrices: {:.2f}s (shape: {})\n".format(time.time() - start_time,
                                                                                              K_test.shape))
        clf = KRR()
        clf.fitted = True
        clf.alpha_ = w_opt
        y_test_pred_idx = clf.predict(K_test[0])
        acc = (y_test_pred_idx == y_test.argmax(dim=1)).sum() / y_test.shape[0]
        rmse = np.sqrt(torch.sum((y_test_pred_idx - y_test.argmax(dim=1)) ** 2) / y_test.shape[0])
        rmse_l.append(rmse.item())
        mean_rmse = sum(rmse_l) / len(rmse_l)
    return mean_rmse, acc.item(), y_test_pred_idx


def rbf_training(X, y, X_val, y_val, X_test, y_test, epoch, v):
    hp_kernel = torch.logspace(-3, 0, 14)  # gamma的取值空间，带宽！！！
    p_lambda = 25  ###
    hp_params = torch.logspace(-8, 2, p_lambda)  # lambd的取值空间，正则化强度！！！
    k_class = K_RBF()
    rmse_l = []

    for e in range(epoch):
        if v: print("RBF epoch {}/{}".format(e + 1, epoch))
        # 计算训练集的Gram矩阵
        K_train = k_class.compute_gram_matrix(X)
        # 计算验证集与训练集之间的交叉Gram矩阵
        K_val = k_class.compute_cross_gram_matrix(X, X_val)

        _, _, hps, w_opt = train_multiple_kernels(K_train,
                                                  K_val,
                                                  y,
                                                  y_val,
                                                  hp_params,
                                                  hp_kernel,
                                                  subsample=1)
        gammas = torch.tensor(hps["KRR"]["hp_kernel"])
        # print("Gammas rbf", gammas)
        start_time = time.time()
        K_test = k_class.compute_cross_gram_matrix(X, X_test, gammas)
        print('K_test的形状:', K_test.shape)
        if v: print("time elapsed computing both Gram matrices {:.2f}s (shape: {})\n".format(time.time() - start_time,
                                                                                             K_test.shape))
        # 后续可以添加使用Gram矩阵进行模型训练、预测等操作，例如使用KRR模型
        clf = KRR()  # 初始化KRR模型（可能需要根据实际情况传入参数）
        clf.fitted = True
        clf.alpha_ = w_opt
        # print('K_train.shape[-1]', K_train.shape[-1])
        # print('len(y):', len(y))
        # clf.fit(K_train, y)  # 使用训练集的Gram矩阵和标签进行模型训练
        y_test_pred_idx = clf.predict(K_test[0])  # 使用测试集的Gram矩阵进行预测
        acc = (y_test_pred_idx == y_test.argmax(dim=1)).sum() / y_test.shape[0]
        rmse = np.sqrt(torch.sum((y_test_pred_idx - y_test.argmax(dim=1)) ** 2) / y_test.shape[0])
        rmse_l.append(rmse.item())
        mean_rmse = sum(rmse_l) / len(rmse_l)
    return mean_rmse, acc.item(), y_test_pred_idx