# import torch
# import time
#
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#
# class K_RBF():
#     def rbf_kernel_torch(x1, x2, hp_kernel):
#         """
#         使用PyTorch计算RBF核函数的值
#         Args:
#             x1 (Tensor): 形状为(n, d)的张量，表示输入数据
#             x2 (Tensor): 形状为(m, d)的张量，表示输入数据
#         Returns:
#             Tensor: 形状为(n, m)的张量，表示RBF核函数的值
#         """
#         # 使用PyTorch计算欧几里得距离
#         x1 = x1.squeeze()
#         x2 = x2.squeeze()
#         distance = torch.norm(x1 - x2, dim=-1)
#         # print('x1的形状', x1.shape)
#         # print('x2的形状', x2.shape)
#         return torch.exp(-(distance ** 2) / (2 * (hp_kernel ** 2)))
#
#     def compute_gram_matrix(X, rbf_kernel_torch, hp_kernel):
#         """
#         计算输入数据X的Gram矩阵
#         Args:
#             X (Tensor): 形状为(T, epoch, n, d)的张量，表示输入数据
#             rbf_kernel_torch (function): RBF核函数计算方法
#             hp_kernel (float): RBF核函数的超参数
#         Returns:
#             Tensor: 形状为(hp_kernel, T, T)的Gram矩阵
#         """
#         T, epoch, n, d = X.shape
#         # print('X的形状:',X.shape)
#         G = torch.zeros((len(hp_kernel), T, T))
#         for i, sigma in enumerate(hp_kernel):
#             for t in range(T):
#                 for s in range(t + 1, T):
#                     # 展平后计算Gram矩阵
#                     Xt_flat = X[t].view(-1, d)
#                     Xs_flat = X[s].view(-1, d)
#                     G[i, t, s] = rbf_kernel_torch(Xt_flat, Xs_flat, sigma).mean()
#             G[i] = G[i] + G[i].t()
#             for t in range(T):
#                 Xt_flat = X[t].view(-1, d)
#                 G[i, t, t] = rbf_kernel_torch(Xt_flat, Xt_flat, sigma).mean()
#         # print('len(G):',len(G))
#         return G
#
#     def compute_cross_gram_matrix(X1, X2, rbf_kernel_torch, hp_kernel):
#         """
#         计算输入数据X1和X2之间的交叉Gram矩阵
#         Args:
#             X1 (Tensor): 形状为(T1, epoch, n, d)的张量
#             X2 (Tensor): 形状为(T2, epoch, n, d)的张量
#             rbf_kernel_torch (function): RBF核函数计算方法
#             hp_kernel (float): RBF核函数的超参数
#         Returns:
#             Tensor: 形状为(hp_kernel, T2, T1)的交叉Gram矩阵
#         """
#         T1, epoch1, n1, d1 = X1.shape
#         # print('X1的形状:',X1.shape)
#         T2, epoch2, n2, d2 = X2.shape
#         assert d1 == d2, "输入特征维度必须匹配"
#         G = torch.zeros((len(hp_kernel), T2, T1))
#         for i, sigma in enumerate(hp_kernel):
#             for t1 in range(T1):
#                 for t2 in range(T2):
#                     X1_flat = X1[t1].view(-1, d1)
#                     X2_flat = X2[t2].view(-1, d2)
#                     G[i, t2, t1] = rbf_kernel_torch(X1_flat, X2_flat, sigma).mean()
#         # print('len(G)cross:',len(G))
#         return G
