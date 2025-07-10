import csv
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# ———————————————————————————————————————————————————————————结果记录————————————————————————————————————————————————————

# 创建文件夹(程序启动时创建)
def createdir():
    # 创建文件夹
    root = 'recording/Acc&Loss'
    timrstr = time.strftime("%Y%m%d-%H%M%S")
    dir_name = ''
    if os.path.exists('./recording/Acc&Loss/timrstr'):
        dir_name = timrstr + '_re'
    else:
        dir_name = timrstr
    dir_root = os.path.join(root, dir_name)
    os.mkdir(dir_root)

    return dir_root, timrstr



def save_parameter_tocsv(time, model, dataset, overlap, segment,
                         random_seed, max_epoch, batch_size, learning_rate, dropout, LS_rate, train_rate,
                         graph_type, window_size, stride, out_gcn, out_gru, change):
    record_dir = './recording'

    with open(os.path.join(record_dir, '实验结果.csv'), 'a', encoding='utf-8', newline='') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow((time, model, dataset, overlap, segment,
                         random_seed, max_epoch, batch_size, learning_rate, dropout, LS_rate, train_rate,
                         graph_type, window_size, stride, out_gcn, out_gru, change))



def save_result_tocsv(test_mAcc, test_mStd, test_mSEN, test_mSPE, test_mF1, test_mKappa, val_mAcc, val_mF1, val_mStd):
    record_dir = './recording'
    csv_file = os.path.join(record_dir, '实验结果.csv')

    # 读取CSV文件
    df = pd.read_csv(csv_file)
    # 获取上一行的数据
    last_row = df.iloc[-1]
    # 获取上一行的最后9列的列名
    columns = last_row.index[-9:]
    # 填充数据
    last_row[columns] = [round(test_mAcc, 4), round(test_mStd, 4), round(test_mSEN, 4), round(test_mSPE, 4), round(test_mF1, 4), round(test_mKappa, 4), round(val_mAcc, 4), round(val_mF1, 4), round(val_mStd, 4)]
    # 将修改后的数据写回CSV文件
    df.iloc[-1] = last_row
    df.to_csv(csv_file, index=False)
# ———————————————————————————————————————————————————————————结果记录————————————————————————————————————————————————————





# ———————————————————————————————————————————————————————————图像绘制————————————————————————————————————————————————————

def pltAcc_TV_nfold(dir, out_fold, train, valid):
    plt.clf()
    plt.plot(train)
    plt.plot(valid)
    plt.title('Fold{} Model accuracy'.format(out_fold+1))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')

    # 每隔20个单位绘制一条垂直线
    for x in range(0, len(train)+1, 20):
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5)

    # 每隔0.05绘制一条水平线
    for y in np.arange(0.60, 1.01, 0.1): # 0.75, 1.01, 0.05          0.825, 1.01, 0.025
        plt.axhline(y=y, color='gray', linestyle='--', alpha=0.5)

    # plt.show()
    plt.savefig(os.path.join(dir, 'Acc_TV_fold{}.png'.format(out_fold+1)))



def pltLoss_TV_nfold(dir, out_fold, train, valid):
    plt.clf()
    plt.plot(train)
    plt.plot(valid)
    plt.title('Fold{} Model loss'.format(out_fold+1))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')

    # 每隔20个单位绘制一条垂直线
    for x in range(0, len(train)+1, 20):
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5)

    # 每隔0.05绘制一条水平线
    for y in np.arange(0.19, 0.71, 0.1): # 0.2, 0.61, 0.05          0.2, 0.51, 0.05
        plt.axhline(y=y, color='gray', linestyle='--', alpha=0.5)

    # plt.show()
    plt.savefig(os.path.join(dir, 'Loss_TV_fold{}.png'.format(out_fold+1)))



def pltAcc_TV_individual(dir, train, valid):
    plt.clf()
    plt.plot(train)
    plt.plot(valid)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    # plt.show()
    plt.savefig(os.path.join(dir, 'Acc_TV_individual.png'))



def pltLoss_TV_individual(dir, train, valid):
    plt.clf()
    plt.plot(train)
    plt.plot(valid)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    # plt.show()
    plt.savefig(os.path.join(dir, 'Loss_TV_individual.png'))



def pltROC(dir, y_true, y_scores):
    plt.clf()
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.3f})'.format(roc_auc))
    # plt.plot(fpr, tpr, color='blue')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')

    # 每隔0.2绘制一条垂直线
    for x in np.arange(0, 1.01, 0.2):
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
    for y in np.arange(0, 1.01, 0.2):
        plt.axhline(y=y, color='gray', linestyle='--', alpha=0.5)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate (Sensitivity)')
    plt.ylabel('True Positive Rate (1 - Specificity)')
    plt.legend(loc='lower right')
    plt.show()
    # plt.savefig(os.path.join(dir, 'ROC.png'))



def plt_ConfusionMatrix(cm, mode="other", normalize=True):
    plt.clf()
    cm_old = cm.copy()  # 保留原始计数值

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    else:
        pass

    plt.imshow(cm, interpolation='nearest', cmap='YlOrBr')
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.rcParams['font.size'] = 16
    plt.colorbar()

    class_names = ['MCI', 'HC']
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=14)
    plt.yticks(tick_marks, class_names, fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.ylim(len(class_names) - 0.5, -0.5)

    # fmt = '.4f' if normalize else 'd'

    # 添加数值标签
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm_old[i, j])  # 原始个数
            percent = cm[i, j] * 100  # 百分比

            # 判断正确或错误的类别
            if i == j:
                label = "True Pos" if i == 0 else "True Neg"  # 对角线 (正确分类)
            else:
                label = "False Pos" if i == 0 else "False Neg"  # 非对角线 (错误分类)

            plt.text(j, i, f"{label}\n{count}\n{percent:.2f}%",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.show()

    # 保存混淆矩阵数据到 CSV 文件
    cm_values = cm_old.flatten()  # 展开成 1D 数组 [TP, FP, FN, TN]
    df = pd.DataFrame([cm_values], columns=["TP", "FP", "FN", "TN"])

    # 如果文件不存在，写入表头，否则追加
    if not os.path.exists("./recording/混淆矩阵.csv"):
        df.to_csv("./recording/混淆矩阵.csv", index=False, mode='w')  # 写入新文件（带表头）
    else:
        df.to_csv("./recording/混淆矩阵.csv", index=False, mode='a', header=False)  # 追加数据（不写表头）

# ———————————————————————————————————————————————————————————图像绘制————————————————————————————————————————————————————