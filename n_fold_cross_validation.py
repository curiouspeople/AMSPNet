import numpy as np
import datetime
import os
import csv
import h5py
import copy
import time
import pickle
import os.path as osp

from toolbox import *
from train_model import *
from util import Averager, ensure_path
from sklearn.model_selection import KFold



# 设置工作根目录
ROOT = os.getcwd()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class NFoldCrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.dir_root = None
        self.Record = args.record

        # 记录结果
        result_path = osp.join(args.save_path, 'result')
        ensure_path(result_path)

        # 保存结果文件的名称（此处为result_EEG.txt）
        self.text_file = osp.join(result_path, "results_{}.txt".format(args.dataset))   # 此处注意服务器与本地区别

        if self.Record:
            # 创建保存结果的文件夹的地址
            self.dir_root, timrstr = createdir()

            file = open(self.text_file, 'a')
            # 下面为txt文件内记录的结果

            file.write("\n" + str(timrstr) +
                       "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
                       "\n######## Data ########" +
                       "\n1)overlap:" + str(args.overlap) +
                       "\n2)segment:" + str(args.segment) +

                       "\n######## Training Process ########" +
                       "\n1)random_seed:" + str(args.random_seed) +
                       "\n2)num_epochs:" + str(args.max_epoch) +
                       "\n3)batch_size:" + str(args.batch_size) +
                       "\n4)learning_rate:" + str(args.learning_rate) +
                       "\n5)dropout:" + str(args.dropout) +
                       "\n6)ls_rate:" + str(args.LS_rate) +
                       "\n7)train_rate:" + str(args.train_rate) +

                       "\n######## Model Parameters ########" +
                       "\n1)graph-type:" + str(args.graph_type) +
                       "\n2)window_size:" + str(args.window_size) +
                       "\n3)stride:" + str(args.stride) +
                       "\n4)out_gcn:" + str(args.out_gcn) +
                       "\n5)out_gru:" + str(args.out_gru) +

                       "\n######## Change last time ######" +
                       "\nchange:" + str(args.change) +
                       "\n######## Result ########" + '\n')

            file.close()

            save_parameter_tocsv(timrstr, args.model, args.dataset, args.overlap, args.segment,
                                 args.random_seed, args.max_epoch, args.batch_size, args.learning_rate,
                                 args.dropout, args.LS_rate, args.train_rate,
                                 args.graph_type, args.window_size, args.stride, args.out_gcn, args.out_gru,
                                 args.change)


    def load_subject_data(self, sub):
        """
        加载被试的数据
        :param sub: 加载哪个被试
        :return: data and label
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = osp.join(save_path, data_type, sub_code)
        dataset = h5py.File(path, 'r')
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
        # print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label


    def prepare_data(self, idx_train, idx_test, data, label):
        """
        1. 根据索引获取训练和测试数据
        2. numpy.array-->torch.tensor
        :param idx_train: index of training data
        :param idx_test: index of testing data
        :param data: (segments, 1, channel, data)
        :param label: (segments,)
        :return: data and label
        """
        data_train = data[idx_train]
        label_train = label[idx_train]
        data_test = data[idx_test]
        label_test = label[idx_test]

        # 归一化
        data_train, data_test = self.normalize(train=data_train, test=data_test)

        # 使用PyTorch为模型训练准备数据
        # 转换为PyTorch张量，并将其数据类型设置为float
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).long()

        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()

        return data_train, label_train, data_test, label_test


    def normalize(self, train, test):
        """
        该函数对脑电信号逐个通道进行标准归一化
        :param train: training data (num_segment, 1, channel, data)
        :param test: testing data (num_segment, 1, channel, data)
        :return: 归一化后的training and testing data
        """
        # data: num_segment * 1 * channel * data
        for channel in range(train.shape[2]):
            # 在每个通道上计算均值和标准差
            mean = np.mean(train[:, :, channel, :])
            std = np.std(train[:, :, channel, :])
            # 将每个通道的数据减去均值，然后除以标准差
            train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
            test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
        return train, test


    def split_balance_class(self, data, label, train_rate, random):
        """
        使用两个类样本的相同百分比获取验证集
        :param data: training data (num_segment, 1, channel, data)
        :param label: (num_segment,)
        :param train_rate: 训练数据的百分比
        :param random(bool): 是否在得到验证数据前对训练数据进行洗牌
        :return: data_trian, label_train, and data_val, label_val
        """
        # Data dimension: (num_segment, 1, channel, data)
        # Label dimension: (num_segment, 1)
        np.random.seed(0)
        # data : (num_segment, 1, channel, data)
        # label : (num_segment,)

        index_0 = np.where(label == 0)[0]   # 类别0的索引
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_random_0 = copy.deepcopy(index_0)

        # for class 1
        index_random_1 = copy.deepcopy(index_1)

        # 进行随机洗牌
        if random == True:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)

        # 根据train_rate计算训练集和验证集
        # 获取训练集和验证集的索引
        index_train = np.concatenate((index_random_0[:int(len(index_random_0) * train_rate)], index_random_1[:int(len(index_random_1) * train_rate)]), axis=0)
        index_val = np.concatenate((index_random_0[int(len(index_random_0) * train_rate):], index_random_1[int(len(index_random_1) * train_rate):]), axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label


    def n_fold_CV(self, subject=[0], fold=5, shuffle=True): # fold: 10 ——> 5
        """
        这个函数实现了n折交叉验证
        :param subject: 加载的被试数量
        :param fold: 折数
        """
        # 初始化输出指标
        tta = []    # total test accuracy
        ttf = []    # total test f1
        ttse = []   # total test sensitivity
        ttsp = []   # total test specificity
        ttk = []    # total test kappa

        tva = []  # total valid accuracy
        tvf = []  # total valid f1

        data = None
        label = None

        # 读取所有被试的数据和标签
        for sub in subject:
            data_sub, label_sub = self.load_subject_data(sub)
            # print('被试{}的数据形状{}， 标签形状{}'.format(sub + 1, data_sub.shape, label_sub.shape))
            if data is None:
                data = np.empty(data_sub.shape)
                label = np.empty(label_sub.shape)
                data[:] = data_sub
                label[:] = label_sub
            data = np.concatenate((data, data_sub), axis=0)
            label = np.concatenate((label, label_sub), axis=0)
        print('all sub data shape:{}， label shape:{}'.format(data.shape, label.shape))

        # 实例化 验证集平均acc和f1-score
        va_val = Averager()
        vf_val = Averager()

        # 初始化测试集的预测值和实际值的列表
        kf = KFold(n_splits=fold, shuffle=shuffle)
        start_time = time.time()    # 记录开始时间

        # n折交叉验证开始
        for idx_fold, (idx_train, idx_test) in enumerate(kf.split(data)):
            print('-----------------------------------------------------')
            print('-----------------------------------------------------')
            print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold+1))

            # 获取训练集，测试集的数据和标签
            data_train, label_train, data_test, label_test = self.prepare_data(idx_train=idx_train, idx_test=idx_test, data=data, label=label)

            # ————————————————————————————————————————————————————此处用于复现论文中的准确率——————————————————————————————————————————————————————
            if self.args.reproduce:     # 此处self.args.reproduce:False
                acc_test, pred, act = test(args=self.args, data=data_test, label=label_test, reproduce=self.args.reproduce, fold=idx_fold)
                acc_val = 0
                f1_val = 0
            # ————————————————————————————————————————————————————此处用于复现论文中的准确率——————————————————————————————————————————————————————

            # 不复现论文中的准确率
            else:
                # 训练新模型
                # 获取验证集的acc和f1-score
                acc_val, f1_val = self.first_stage(data=data_train, label=label_train, fold=idx_fold)

                #获取测试集的acc，预测值与实际值
                _, pred, act = test(args=self.args, data=data_test, label=label_test, reproduce=self.args.reproduce, fold=idx_fold)

            # 最终输出的验证集的acc和f1-score
            va_val.add(acc_val)
            vf_val.add(f1_val)
            tva.append(va_val.item())   # 验证集acc
            tvf.append(vf_val.item())   # 验证集f1-score

            # 最终输出的测试集的acc和f1-score
            acc_perfold, sen_perfold, spe_perfold, f1_perfold, kappa_perfold, cm_perfold = get_metrics(y_pred=pred, y_true=act)
            tta.append(acc_perfold)         # 测试集acc
            ttse.append(sen_perfold)        # 测试集sen
            ttsp.append(spe_perfold)        # 测试集spe
            ttf.append(f1_perfold)          # 测试集f1
            ttk.append(kappa_perfold)       # 测试集kappa
            # ttc.append(cm)          # 测试集cm

            if self.args.record:
                # 记录每一折的结果
                result = 'Fold {}, test acc = {:.4f}, test f1 = {:.4f}'.format(idx_fold+1, tta[-1], ttf[-1])
                self.log2txt(result)

        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时长（秒）

        # 最终结果显示
        tta = np.array(tta)     # total test accuracy
        ttse = np.array(ttse)   # total test sensitivity
        ttsp = np.array(ttsp)   # total test specifcity
        ttf = np.array(ttf)     # total test f1
        ttk = np.array(ttk)     # total test kappa

        tva = np.array(tva)     # total valid accuracy
        tvf = np.array(tvf)     # total valid f1

        mACC = np.mean(tta)     # 测试集平均准确率
        mstd = np.std(tta)      # 测试集准确率标准差
        mSEN  = np.mean(ttse)   # 测试集平均敏感性
        mSPE = np.mean(ttsp)    # 测试集平均特异性
        mF1 = np.mean(ttf)      # 测试集平均f1-score
        mKap = np.mean(ttk)     # 测试集平均kappa分数

        mACC_val = np.mean(tva) # 验证集平均准确率
        mF1_val = np.mean(tvf)  # 验证集平均f1-score
        std_val = np.std(tva)   # 验证集平均标准差

        print('Final: test mean ACC:{:.4f} std:{:.4f}'.format(mACC, mstd * 100))
        print('Final: test mean SEN:{:.4f}'.format(mSEN))
        print('Final: test mean SPE:{:.4f}'.format(mSPE))
        print('Final: test mean F1:{:.4f}'.format(mF1))
        print('Final: test mean Kappa:{:.4f}'.format(mKap))

        print('Final: val mean ACC:{:.4f} std:{:.4f}'.format(mACC_val, std_val * 100))
        print('Final: val mean F1:{:.4f}'.format(mF1_val))
        print('Final: train time :{:.2f}s'.format(total_time))

        if self.args.record:
            results = 'test mAcc = {:.4f} mStd = {:.4f} mSEN = {:.4f} mSPE = {:.4f} mF1 = {:.4f} mKappa = {:.4f}    val mAcc = {:.4f} mStd = {:.4f} mF1 = {:.4f}'.format(mACC, mstd * 100, mSEN, mSPE, mF1, mKap, mACC_val, std_val * 100, mF1_val)
            self.log2txt(results)

            # 将参数保存到csv文件中
            save_result_tocsv(mACC, mstd * 100, mSEN, mSPE, mF1, mKap, mACC_val, std_val * 100, mF1_val)


    def first_stage(self, data, label, fold):
        """
        此函数first_stage为了:
            1. 在训练数据上选择超参数
            2. 获得测试数据评估的模型
            3. 保存阶段1的训练集与测试集的准确率图像与损失图像
        :return: 验证集平均acc 和 验证集平均f1-score
        """
        # data: (num_segments, 1, channel*data)
        # 实例化验证集平均acc和f1-score
        va = Averager()
        vf = Averager()

        va_item = []
        maxAcc = 0.0

        data_train, label_train, data_val, label_val = self.split_balance_class(data, label, self.args.train_rate, random=True)
        print('-----------------------------------------------------')
        # 获取验证集的最高acc及其对应f1-score
        acc_val, F1_val, total_params = train(args=self.args,
                                             dir=self.dir_root,
                                             data_train=data_train,
                                             label_train=label_train,
                                             data_val=data_val,
                                             label_val=label_val,
                                             out_fold=fold)

        va.add(acc_val)
        vf.add(F1_val)
        va_item.append(acc_val)

        if acc_val >= maxAcc:
            # 选择值acc较高的模型作为第二阶段的模型
            old_name = osp.join(self.args.save_path, 'candidate.pth')
            new_name = osp.join(self.args.save_path, 'final_model.pth')
            if os.path.exists(new_name):
                os.remove(new_name)
            os.rename(old_name, new_name)
            print('New max ACC model saved, with the val ACC being:{}'.format(acc_val))
            print('-----------------------------------------------------')

            if self.args.record:
                # 记录每一折的结果
                result = 'Fold {}, Val acc = {:.4f}, Total params = {}'.format(fold + 1, acc_val, total_params)
                self.log2txt(result)

        mAcc = va.item()
        mF1 = vf.item()
        return mAcc, mF1


    def log2txt(self, content):
        """
        该函数将内容记录到results.txt中
        :param content: string, the content to log
        """
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()
