import torch
import pandas as pd
import pickle
import time
import torch.nn.functional as F
import numpy as np
import scipy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.synthetic_helpers import *
# from src.RBF_synthetic import *
from src.EBSW_kernels import k_ebsw_rf
from src.MMDkernels import k_MMD
from src.training import train_multiple_kernels
from src.classifiers import KRR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc
from src.outjoin import crop_and_pad_images
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from src.outjoin import rotate_images
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

CutMnist_all_fpr = pd.DataFrame()

print('MNIST-(1500,600,1000;L=1500,e=1)EBSW')

num = 1
for x in range(num):
    print('第{}次循环:'.format(x))
    # MNIST 设置
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 加载 MNIST 数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    # val_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_size = len(train_dataset) // 2
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # 创建数据加载器
    dataloader_train = DataLoader(train_subset, batch_size=1500, shuffle=True)#true试试
    dataloader_val = DataLoader(val_subset, batch_size=600, shuffle=True)#true试试
    dataloader_test = DataLoader(val_dataset, batch_size=1000, shuffle=True)#不影响

    # 提取训练、验证和测试的数据子集
    X, y = next(iter(dataloader_train))
    X_val, y_val = next(iter(dataloader_val))
    X_test, y_test = next(iter(dataloader_test))

    # 将标签 y 转换为独点编码的格式,假设有10个类别（MNIST 的标签是从 0 到 9）
    num_classes = 10
    y = F.one_hot(y, num_classes=num_classes).float()
    y_val = F.one_hot(y_val, num_classes=num_classes).float()
    y_test = F.one_hot(y_test, num_classes=num_classes).float()

    # 将图像展平并调整为特定的格式以用于核方法,格式调整为 (batch_size, epoch, channels, height, width)
    epoch = 1  # 训练的轮数
    X = X.unsqueeze(1).repeat(1, epoch, 1, 1, 1)
    X_val = X_val.unsqueeze(1).repeat(1, epoch, 1, 1, 1)
    X_test = X_test.unsqueeze(1).repeat(1, epoch, 1, 1, 1)

    # 将图像展平为 (1, 784) 的向量以便后续处理
    X = X.view(X.size(0), epoch, 1, -1)
    X_val = X_val.view(X_val.size(0), epoch, 1, -1)
    X_test = X_test.view(X_test.size(0), epoch, 1, -1)

    # #假设我们裁剪掉顶部和底部 4 行，左右各 4 列
    # X_test_cropped_padded = crop_and_pad_images(X_test, crop_top=4, crop_bottom=4, crop_left=4, crop_right=4)
    # X_test_cropped_padded = X_test_cropped_padded.view(X_test_cropped_padded.size(0), epoch, 1, -1)

    # # 设置旋转度数
    # rotation_angle = 45
    # X_test_rotated = rotate_images(X_test, rotation_angle)
    # # 展平旋转后的测试集
    # X_test_rotated = X_test_rotated.view(X_test_rotated.size(0), epoch, 1, -1)

    # 参数设置
    v = True  # 是否详细输出的标志
    dic_res = {"EBSW2": [], "EBSW1": [], "MMD": [], "RBF": []}
    dic_acc = {"EBSW2": [], "EBSW1": [], "MMD": [], "RBF": []}
    dic_conf_matrix = {"EBSW2": [], "EBSW1": [], "MMD": [], "RBF": []}
    dic_recall = {"EBSW2": [], "EBSW1": [], "MMD": [], "RBF": []}
    dic_precision = {"EBSW2": [], "EBSW1": [], "MMD": [], "RBF": []}
    dic_f1 = {"EBSW2": [], "EBSW1": [], "MMD": [], "RBF": []}

    #如果启用了 EBSW2，则使用 EBSW2 核训练
    if "EBSW2" in dic_res:
        print("开始使用 EBSW2 进行训练，训练轮数为 {}".format(epoch))
        rmse, acc, y_test_pred_idx = EBSW_training(X, y, X_val, y_val, X_test, y_test, 2, epoch, v)
        print("acc:", acc)
        conf_matrix = confusion_matrix(y_test_pred_idx, y_test.argmax(dim=1))
        accuracy = accuracy_score(y_test_pred_idx, y_test.argmax(dim=1))
        recall = recall_score(y_test_pred_idx, y_test.argmax(dim=1), average='macro')
        precision = precision_score(y_test_pred_idx, y_test.argmax(dim=1), average='macro')
        f1 = f1_score(y_test_pred_idx, y_test.argmax(dim=1), average='macro')

        dic_res["EBSW2"].append(rmse)
        dic_conf_matrix["EBSW2"].append(conf_matrix)
        dic_acc["EBSW2"].append(acc)
        dic_recall["EBSW2"].append(recall)
        dic_precision["EBSW2"].append(precision)
        dic_f1["EBSW2"].append(f1)

        # print("混淆矩阵:")
        # print(conf_matrix)
        # print("准确率: ", accuracy)
        # print("召回率: ", recall)
        # print("精确率: ", precision)
        # print("F1-score: ", f1)

    # 如果启用了 EBSW1，则使用 EBSW1 核训练
    if "EBSW1" in dic_res:
        print("开始使用 EBSW1 进行训练，训练轮数为 {}".format(epoch))
        rmse, acc, y_test_pred_idx = EBSW_training(X, y, X_val, y_val, X_test, y_test, 1, epoch,v)
        print('acc:', acc)
        conf_matrix = confusion_matrix(y_test_pred_idx, y_test.argmax(dim=1))
        accuracy = accuracy_score(y_test_pred_idx, y_test.argmax(dim=1))
        recall = recall_score(y_test_pred_idx, y_test.argmax(dim=1), average='macro')
        precision = precision_score(y_test_pred_idx, y_test.argmax(dim=1), average='macro')
        f1 = f1_score(y_test_pred_idx, y_test.argmax(dim=1), average='macro')

        dic_res["EBSW1"].append(rmse)
        dic_conf_matrix["EBSW1"].append(conf_matrix)
        dic_acc["EBSW1"].append(acc)
        dic_recall["EBSW1"].append(recall)
        dic_precision["EBSW1"].append(precision)
        dic_f1["EBSW1"].append(f1)

        # print("混淆矩阵:")
        # print(conf_matrix)
        # print("准确率: ", accuracy)
        # print("召回率: ", recall)
        # print("精确率: ", precision)
        # print("F1-score: ", f1)
    if "RBF" in dic_res:
        print("开始使用RBF核函数进行训练，训练轮数为{}".format(epoch))
        rmse, acc, y_test_pred_idx = rbf_training(X, y, X_val, y_val, X_test, y_test, epoch, v)
        print('acc:',acc)
        #将y_pred和y_test.argmax(dim=1)移动到CPU上
        y_test_pred_idx = y_test_pred_idx.cpu()
        y_test_argmax = y_test.argmax(dim=1).cpu()
        conf_matrix = confusion_matrix(y_test_pred_idx, y_test.argmax(dim=1))
        accuracy = accuracy_score(y_test_pred_idx, y_test.argmax(dim=1))
        recall = recall_score(y_test_pred_idx, y_test.argmax(dim=1), average='macro')
        precision = precision_score(y_test_pred_idx, y_test.argmax(dim=1), average='macro')
        f1 = f1_score(y_test_pred_idx, y_test.argmax(dim=1), average='macro')

        dic_res["RBF"].append(rmse)
        dic_conf_matrix["RBF"].append(conf_matrix)
        dic_acc["RBF"].append(acc)
        dic_recall["RBF"].append(recall)
        dic_precision["RBF"].append(precision)
        dic_f1["RBF"].append(f1)

        # print("混淆矩阵:")
        # print(conf_matrix)
        # print("准确率: ", accuracy)
        # print("召回率: ", recall)
        # print("精确率: ", precision)
        # print("F1-score: ", f1)

    if "MMD" in dic_res:
        print("开始使用 MMD 进行训练，训练轮数为 {}".format(epoch))
        # 计算并输出混淆矩阵、准确率、召回率、精确率、F1-score
        rmse, acc ,y_test_pred_idx = mmd_training(X, y, X_val, y_val, X_test, y_test, epoch, v)  # 假设这里有一个预测函数来获取预测结果
        print('acc',acc)
        #将y_pred和y_test.argmax(dim=1)移动到CPU上
        y_pred = y_test_pred_idx.cpu()
        y_test_argmax = y_test.argmax(dim=1).cpu()
        conf_matrix = confusion_matrix(y_pred, y_test_argmax)
        accuracy = accuracy_score(y_pred, y_test_argmax)
        recall = recall_score(y_pred, y_test_argmax, average='macro')
        precision = precision_score(y_pred, y_test_argmax, average='macro')
        f1 = f1_score(y_pred, y_test_argmax, average='macro')

        dic_res["MMD"].append(rmse)
        dic_conf_matrix["MMD"].append(conf_matrix)
        dic_acc["MMD"].append(acc)
        dic_recall["MMD"].append(recall)
        dic_precision["MMD"].append(precision)
        dic_f1["MMD"].append(f1)

        # print("混淆矩阵:")
        # print(conf_matrix)
        # print("准确率: ", accuracy)
        # print("召回率: ", recall)
        # print("精确率: ", precision)
        # print("F1-score: ", f1)

    temp_df = pd.DataFrame()
    temp_df['迭代次数'] = [x + 1]
    if "EBSW2" in dic_res:
        temp_df['EBSW2_res'] = dic_res["EBSW2"]
        temp_df['EBSW2_acc'] = dic_acc["EBSW2"]
        temp_df['EBSW2_recall'] = dic_recall["EBSW2"]
        temp_df['EBSW2_precision'] = dic_precision["EBSW2"]
        temp_df['EBSW2_f1'] = dic_f1["EBSW2"]
        temp_df['EBSW2_conf_matrix'] = dic_conf_matrix["EBSW2"]
    if "EBSW1" in dic_res:
        temp_df['EBSW1_res'] = dic_res["EBSW1"]
        temp_df['EBSW1_acc'] = dic_acc["EBSW1"]
        temp_df['EBSW1_recall'] = dic_recall["EBSW1"]
        temp_df['EBSW1_precision'] = dic_precision["EBSW1"]
        temp_df['EBSW1_f1'] = dic_f1["EBSW1"]
        temp_df['EBSW1_conf_matrix'] = dic_conf_matrix["EBSW1"]
    if "RBF" in dic_res:
        temp_df['RBF_res'] = dic_res["RBF"]
        temp_df['RBF_acc'] = dic_acc["RBF"]
        temp_df['RBF_recall'] = dic_recall["RBF"]
        temp_df['RBF_precision'] = dic_precision["RBF"]
        temp_df['RBF_f1'] = dic_f1["RBF"]
        temp_df['RBF_conf_matrix'] = dic_conf_matrix["RBF"]
    if "MMD" in dic_res:
        temp_df['MMD_res'] = dic_res["MMD"]
        temp_df['MMD_acc'] = dic_acc["MMD"]
        temp_df['MMD_recall'] = dic_recall["MMD"]
        temp_df['MMD_precision'] = dic_precision["MMD"]
        temp_df['MMD_f1'] = dic_f1["MMD"]
        temp_df['MMD_conf_matrix'] = dic_conf_matrix["MMD"]
    CutMnist_all_fpr = pd.concat([CutMnist_all_fpr, temp_df], ignore_index=True)
# 将汇总后的所有迭代结果保存到同一个Excel文件
CutMnist_all_fpr.to_excel('CIFRA10-(1500,600,1000;L=1500,e=1)_EBSW.xlsx', index=False)
