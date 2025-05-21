import warnings
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
import sys

class k_ebsw_rf():
    def __init__(self, M=1, r=0, non_uniform=False, d_in=2, deterministic=False, p=2, true_rf=False):
        """Gaussian-Sliced-Wasserstein-Kernel with Random Slices.
        Args:
            M: number of sampled slices
            r: number of point sampled per slice (if r=0, compute the inner 
            Wasserstein distance in closed form)
            non_uniform: if True normalized pixel intensities are used in the
            (Fashion)-Mnist experiments
            d_in: dimension of the inputs (>=2)
            p (int): p-SW distance for p = 1 or 2
            true_rf: if True uses independent samples on (S^{d_in-1}*(0,1))
            else samples several points on (0,1) for each projection direction
        """
        self.M = M
        self.r = r # shoulc be M == r for true_rf(此处存疑！！！！！！)
        self.d_in = d_in
        self.non_uniform = non_uniform
        assert p == 2 or p ==1
        self.p = p
        assert d_in >= 2, 'Dimension of inputs should be higher than 2 to use SW'
        th = MultivariateNormal(torch.zeros(d_in), torch.eye(d_in)).sample((M,)).to(device)
        th = F.normalize(th)  # shape: M * d，，，(此处存疑！！！！！！)
        assert (M, d_in) == th.shape
        self.th = th.to(device) # store slicing directions
        if self.r > 0: # sample points on the real line to evaluate features    
            self.ts = torch.rand(r).to(device)
        n_ts = len(self.ts) if self.r > 0 else self.r
        self.true_rf = true_rf
        if self.true_rf:
            assert self.r == self.M
            #print("Number of random features: {}".format(self.th.shape[0]))
        #else:
        #    print("Number of slices: {}, t's: {}".format(self.th.shape[0], n_ts))

    def _bins_indexes(self, real_numbers, bins):
        """Converts a batch of real numbers to a batch of indexes for the bins
        where the real numbers fall in.
        """
        v = (real_numbers.view(-1, 1) - bins.view(1, -1))
        v[v < 0] = float("Inf")
        return v.min(dim=1)[1]

    def _bins_indexes_vect(self, real_numbers, bins):
        """Batch vectorised version of bins_indexes"""
        a = real_numbers.unsqueeze(-1)
        b = bins.t().unsqueeze(0).unsqueeze(2)
        v = (a - b).squeeze(0).T
        v[v < 0] = float("Inf")
        return v.min(dim=-1)[1]

    def _bins_indexes_nonvect(self, real_numbers, bins):
        """Batch non vectorised version of bins_indexes"""
        a = real_numbers.unsqueeze(-1)
        v = a - bins
        v[v < 0] = float("Inf")
        return v.min(dim=-1)[1]

    def get_features(self,x):
        """Return the feature map phi(x) (see paper)"""
        if self.non_uniform:
            c = x[:,-1]
            x = x[:,:-1]
        n = x.shape[0]
        x_proj, indexes = (self.th @ x.T).sort(dim=-1) # shape: M * n
        assert (self.M, n) == x_proj.shape
        if self.non_uniform:
            c = c[indexes] # reorder weights
            assert (self.M, n) == c.shape
            bins = torch.hstack((torch.zeros((c.shape[0],1)).to(device), torch.cumsum(c, dim=1)[:,:-1])).to(device)
            if self.true_rf:
                idx_x = self._bins_indexes_nonvect(self.ts, bins)
                phi_x = x_proj.gather(1, idx_x.view(-1, 1)).squeeze()
            else:
                idx_x = self._bins_indexes_vect(self.ts, bins)
                temp = [x_proj[i,idx_x[i]].view(1,-1) for i in range(self.M)]
                phi_x = torch.vstack(temp).flatten()
        else:
            bins = torch.arange(n).to(device) / n
            idx_x = self._bins_indexes(self.ts, bins)
            if self.true_rf:
                phi_x = x_proj.gather(1, idx_x.view(-1, 1)).flatten()
                assert phi_x.dim() == 1
                assert self.M == len(phi_x)
            else:
                phi_x = x_proj[:, idx_x].flatten()
                assert self.r * self.M == len(phi_x)
        return phi_x

    def get_features_set(self, X):
        """Compute the features for a batch of inputs"""
        if self.true_rf:
            X_phi = torch.zeros((len(X), self.M)).to(device)
        else:
            X_phi = torch.zeros((len(X), self.r * self.M)).to(device)
        for i,x in enumerate(X):
            X_phi[i] = self.get_features(x.to(device))
        return X_phi

    def get_grams(self, X, Y, gammas=1.):
        """Compute both K(X,Y) and K(X,X) with low computation"""
        T_x, T_y = len(X), len(Y)
        C = self.M if self.true_rf else self.r * self.M
        X_phi = self.get_features_set(X)
        g = len(gammas)
        vect_diff = X_phi.unsqueeze(1) - X_phi
        assert (T_x, T_x, C) == vect_diff.shape
        if self.p == 2:
            vect_norm =ISEBSW(X, X, L=1500, p=2)
        elif self.p == 1:
            vect_norm =ISEBSW(X, X, L=1500, p=1)
        assert (T_x, T_x) == vect_norm.shape
        vect_g = torch.einsum('g, nm->gnm', gammas, vect_norm)
        assert (g, T_x, T_x) == vect_g.shape
        K_xx = torch.exp(-vect_g / C)
        del vect_diff
        del vect_norm
        del vect_g

        Y_phi = self.get_features_set(Y)
        print("X:",X.shape)
        print("Y:",Y.shape)
        print("X_phi:",X_phi.shape)
        print("Y_phi:",Y_phi.shape)
        vect_diff = Y_phi.unsqueeze(1) - X_phi
        print("vect_diff:",vect_diff.shape)
        assert (T_y, T_x, C) == vect_diff.shape
        if self.p == 2:
            vect_norm =ISEBSW(Y, X, L=1500, p=2)
        elif self.p == 1:
            vect_norm =ISEBSW(Y, X, L=1500, p=1)
        print("vect_norm:",vect_norm.shape)
        print("vect_norm:",vect_norm)            

        assert (T_y, T_x) == vect_norm.shape
        vect_g = torch.einsum('g, nm->gnm', gammas, vect_norm)
        print("vect_g:",vect_g.shape)
        assert (g, T_y, T_x) == vect_g.shape
        K_yx = torch.exp(-vect_g / C)
        del vect_diff
        del vect_norm
        del vect_g
        # print('K_xx的纬度:',K_xx.dim())
        # print('K_xx的形状:',K_xx.shape)
        return K_xx, K_yx


    def get_gram(self, X, gammas=1.):
        """Compute the gram matrix between for X."""
        T = len(X)
        C = self.M if self.true_rf else self.r * self.M
        g = len(gammas)
        X_phi = self.get_features_set(X)
        vect_diff = X_phi.unsqueeze(1) - X_phi
        assert (T, T, C) == vect_diff.shape
        if self.p == 2:
            vect_norm =ISEBSW(X, X, L=1500, p=2)
        elif self.p == 1:
            vect_norm =ISEBSW(X, X, L=1500, p=1)
        assert (T, T) == vect_norm.shape
        vect_g = torch.einsum('g, nm->gnm', gammas, vect_norm)
        assert (g, T, T) == vect_g.shape
        K = torch.exp(-vect_g / C)
        return K

    def get_cross_gram(self, X, Y, gammas=1.):
        """Compute the cross gram matrix between X and Y, K(Y,X)."""
        T1, T2 = len(X), len(Y)
        C = self.M if self.true_rf else self.r * self.M
        X_phi = self.get_features_set(X)
        Y_phi = self.get_features_set(Y)
        #print(X_phi.shape, Y_phi.shape)
        g = len(gammas)
        vect_diff = Y_phi.unsqueeze(1) - X_phi
        assert (T2, T1, C) == vect_diff.shape
        if self.p == 2:
            vect_norm =ISEBSW(Y, X, L=1500, p=2)###试试
        elif self.p == 1:
            vect_norm =ISEBSW(Y, X, L=1500, p=1)
        assert (T2, T1) == vect_norm.shape
        vect_g = torch.einsum('g, nm->gnm', gammas, vect_norm)
        assert (g, T2, T1) == vect_g.shape
        K = torch.exp(-vect_g / C)
        return K

    def compute(self, x, y, gammas=1.):
        """Return the value of the kernel between x and y.

        Args:
            x (Tensor): shape (n,d)
            y (Tensor): shape (m,d)
            gammas (Tensor or float): length-scale(s) for the Gaussian kernel

        Returns:
            k(x,y)
        """
        n, d = x.shape
        assert x.dim() == 2, "x should have shape (n,d)"
        assert y.dim() == 2, "y should have shape (m,d)"
        m = y.shape[0]
        assert d == y.shape[1], "Inputs x and y should have the same dimension"
        assert d == self.d_in + self.non_uniform
        if d == 1:
            print("Inputs should have dimension higher than 2 to use SW")
            warnings.warn("Inputs have dimension 1")
            return
        if self.r == 0:
            return self._compute_no_rf(x, y, gammas=gammas)

        # project and sort
        phi_x = self.get_features(x)
        phi_y = self.get_features(y)
        out = (phi_x - phi_y) ** 2
        assert self.r * self.M == len(out)
        out = torch.exp(-out.sum()/self.r/self.M*gammas)
        return out

    def _compute_no_rf(self, x, y, gammas, degrees=0):
        """Computation of the kernel without random features
        (closed form computation of the inner integral).
        """
        # project and sort
        n,m = x.shape[0], y.shape[0]
        x_proj = (self.th @ x.T).sort(dim=-1)[0]  # shape: M * n
        y_proj = (self.th @ y.T).sort(dim=-1)[0]  # shape: M * m
        assert (self.M, n) == x_proj.shape
        assert (self.M, m) == y_proj.shape
        if n != m:
            n = n * m
            x_proj, y_proj = compute_hack(x_proj.unsqueeze(-1), y_proj.unsqueeze(-1))
            assert (self.M, n) == x_proj.shape
            assert (self.M, n) == y_proj.shape
        
        out = (x_proj - y_proj) ** 2  # component-wise difference
        assert (self.M, n) == out.shape
        out = out.sum() / self.M / n
        out = torch.exp(-out * gammas)
        return out

def compute_hack(x, y):
    n, m = x.shape[1], y.shape[1]
    t_m = torch.ones((m,)).view(-1, 1).to(device)
    t_n = torch.ones((n,)).view(-1, 1).to(device)
    x_ext = (x @ t_m.T).flatten(start_dim=1, end_dim=2)
    y_ext = (y @ t_n.T).flatten(start_dim=1, end_dim=2)
    return x_ext, y_ext

def rand_projections(dim, num_projections=1000, device='cpu'):
    # 生成随机投影方向
    projections = torch.randn((num_projections, dim), device=device)
    # 将每个投影向量规范化为单位长度
    # 原代码没有执行归一化操作，新增归一化如下
    projections = F.normalize(projections, dim=1)
    return projections

def one_dimensional_Wasserstein_prod(X, Y, theta, p):
    # X: (n_sets_X, n_samples_X, d)
    # Y: (n_sets_Y, n_samples_Y, d)
    # theta: (L, d)
    # 输出: wasserstein_distance: (n_sets_X, n_sets_Y, L)
    
    # 将 X 和 Y 投影到 theta 上
    X_prod = torch.matmul(X, theta.transpose(0, 1))  # (n_sets_X, n_samples_X, L)
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))  # (n_sets_Y, n_samples_Y, L)
    
    # 对投影结果进行排序
    X_prod_sorted, _ = torch.sort(X_prod, dim=1)  # (n_sets_X, n_samples_X, L)
    Y_prod_sorted, _ = torch.sort(Y_prod, dim=1)  # (n_sets_Y, n_samples_Y, L)
    #########         按照第几个维度排序，dim=1 or 0 ????????            #############
    # 计算排序后的绝对差值并扩展维度以匹配 X 和 Y 的集合数量
    X_prod_sorted = X_prod_sorted.unsqueeze(1)  # (n_sets_X, 1, n_samples_X, L)# ????????
    Y_prod_sorted = Y_prod_sorted.unsqueeze(0)  # (1, n_sets_Y, n_samples_Y, L)
    
    # 计算 Wasserstein 距离
    wasserstein_distance = torch.abs(X_prod_sorted - Y_prod_sorted)  # (n_sets_X, n_sets_Y, n_samples_X, L)
    ###########      p=1时用abs，p=2时用2范数。     ###########
    # 在样本维度上求和并取 p 次幂
    wasserstein_distance = torch.sum(torch.pow(wasserstein_distance, p), dim=2)  # (n_sets_X, n_sets_Y, L)
    
    return wasserstein_distance

def ISEBSW(X, Y, L=10, p=2, device="cpu"):
    # X: (n_sets_X, n_samples_X, d)
    # Y: (n_sets_Y, n_samples_Y, d)
    n_sets_X, _, dim = X.size()
    n_sets_Y, _, _ = Y.size()
    
    # 生成随机投影方向
    theta = rand_projections(dim, L, device)  # (L, d)
    
    # 计算所有集合的 Wasserstein 距离
    wasserstein_distances = one_dimensional_Wasserstein_prod(X, Y, theta, p=p)  # (n_sets_X, n_sets_Y, L)
    # 使用均匀权重计算加权的 Wasserstein 距离
    #weights = torch.full((n_sets_X, n_sets_Y, L), 1.0 / L, device=device)  # (n_sets_X, n_sets_Y, L)
    #########         这里不是能量权重，应该不对吧          #############
    # 使用 softmax 计算每个方向的权重
    weights = torch.softmax(wasserstein_distances, dim=2)  # (n_sets_X, n_sets_Y, L)
    #使用二次函数计算
    #weights = torch.pow(wasserstein_distances, 2)  # (n_sets_X, n_sets_Y, L)
    weights = weights / torch.sum(weights, dim=2, keepdim=True)  # 归一化权重，使得每个方向的权重之和为 1

    # 计算加权后的 Wasserstein 距离和
    sw = torch.sum(weights * wasserstein_distances, dim=2)  # (n_sets_X, n_sets_Y)
    
    # 取 p 次根
    return torch.pow(sw, 1. / p)  # (n_sets_X, n_sets_Y)


