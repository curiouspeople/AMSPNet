# 这是EEG数据集的处理脚本
import os
import mne
import pywt
import h5py
import torch
import numpy as np
import os.path as osp

from train_model import *



class PrepareData:
    def __init__(self, args):
        # 初始化这里的所有参数
        # arg包含参数设置
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.label_type = args.label_type
        self.original_order = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

        # 按脑区划分通道
        self.graph_gen_EEG = [['Fp1', 'F7', 'F3'], # LF
                              ['Fp2', 'F4', 'F8'],  # RF
                              ['T3', 'T5', 'F7'],   # LT 复用
                              ['C3', 'Cz', 'C4'],   # C
                              ['O1', 'O2', 'Pz'],   # RT 复用
                              ['P3', 'Pz', 'P4'],   # P
                              ['T4', 'T6', 'F8']]   # O 复用
        self.graph_type = args.graph_type


    def run(self, subject_list, split, expand):
        """
        Parameters
        ----------
        subject_list: 需要处理的被试
        split: (bool) 是否将一个试验的数据分成更短的片段
        expand: (bool) 是否为CNN添加一个空维度

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """

        # 对每个被试的数据进行处理
        for sub in subject_list:
            # 读取数据
            data_, label_ = self.load_data_per_subject(sub,
                                                       segment_length=self.args.segment,
                                                       overlap=self.args.overlap,
                                                       sampling_rate=self.args.sampling_rate)

            # 扩展维度以适配输入
            if expand:
                # 在倒数第三个维度上添加新的维度(CNNs)
                data_ = np.expand_dims(data_, axis=-3)

            # 划分数据（实际上是标签转为np格式）
            if split:
                data_, label_ = self.split(data=data_,
                                           label=label_)

            print('Data and label prepared!')
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_, label_, sub)


    def load_data_per_subject(self, sub, segment_length, overlap, sampling_rate):
        """
        这个函数装载目标被试的原始文件
        Parameters
        ----------
        sub: 哪个被试需要加载

        Returns
        -------
        data_subject: (num_segment, channel, data)
        label_subject: (num_segment, )
        """
        sub += 1
        if (sub < 12):  # EEG_1:12   EEG_2:19   EEG_12:30
            sub_code = str('mci' + str(sub) + 'raw.fif')
        else:
            sub_code = str('hc' + str(sub) + 'raw.fif')

        # 初始化存储的数据和标签
        data_segment = sampling_rate * segment_length
        data_subject = np.zeros((0, 19, data_segment))
        label_subject = []
        segments = []
        data = []
        label = []

        subject_path = os.path.join(self.data_path, sub_code)
        subject_raw = mne.io.read_raw_fif(subject_path)
        subject_numpy_data = subject_raw.get_data()

        # 对数据进行分段
        step = int(data_segment * (1 - overlap))                        # 设置滑动窗口
        for row in subject_numpy_data:
            row_segments = []
            for j in range(0, len(row) - data_segment + 1, step):
                segment = row[j:j + data_segment]
                row_segments.append(segment)
            segments.append(row_segments)

        segments_array = np.array(segments)
        segments_array_tran = np.transpose(segments_array, (1, 0, 2))   # 分段数据进行转置

        # 对分好段的数据打标签
        data.extend(segments_array_tran)
        getlabel = sub_code[:3]
        if getlabel == "mci":
            label.extend([0] * len(segments_array_tran))
        else:
            label.extend([1] * len(segments_array_tran))

        data = np.array(data)
        n_sample = len(data)                                            # 分段出来的样本数量
        data = data.reshape((n_sample, 19, data_segment))

        # 存储划分好的数据和对应标签
        data_subject = np.vstack((data_subject, data))
        label_subject += label

        # 重新排序EEG通道，构建局部-全局图
        data_subject = self.reorder_channel(data=data_subject, graph=self.graph_type)               # 仅重排序通道数据
        print('Atfer reorder, data:' + str(data_subject.shape) + ' label:' + str(len(label_subject)))
        return data_subject, label_subject


    def reorder_channel(self, data, graph):
        """
        该函数根据不同的图设计重新排序通道
        Parameters
        ----------
        data: (num_segment, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (num_segment, channel, data)
        """
        if graph == 'gen1':
            graph_idx = self.graph_gen1_EEG
        if graph == 'gen2':
            graph_idx = self.graph_gen2_EEG
        if graph == 'gen3':
            graph_idx = self.graph_gen3_EEG
        if graph == 'gen4':
            graph_idx = self.graph_gen4_EEG
        if graph == 'gen5':
            graph_idx = self.graph_gen5_EEG
        if graph == 'hem':
            graph_idx = self.graph_hem_EEG

        # 对输入数据data的通道维度重新排序
        idx = []
        num_chan_local_graph = []

        for i in range(len(graph_idx)):
            num_chan_local_graph.append(len(graph_idx[i]))  # 迭代 graph_idx 列表中的每个元素，将每个元素的长度添加到 num_chan_local_graph 列表，表示第i个本地图中的通道数量

            # 返回chan在self.original_order中的索引
            for chan in graph_idx[i]:
                idx.append(self.original_order.index(chan)) # self.original_order.index(chan)的结果添加到idx列表中

            # 在utils.py中保存本地图中用于构建LGG模型的通道数量
            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph), 'w')
            dataset['data'] = num_chan_local_graph          # 将num_chan_local_graph列表保存到HDF5文件的data数据集中
            dataset.close()
        return data[:, idx, :]                              # 通过使用切片操作，根据idx列表的顺序重新排序输入数据data的通道维度，并返回重新排序后的数据


    def save(self, data, label, sub):
        """
        此函数将处理后的数据保存到目标文件夹中
        Parameters
        ----------
        data: 处理后的数据
        label: 相应的标签
        sub: 被试ID

        Returns
        -------
        None
        """
        save_path = os.getcwd()                     # 获取当前工作目录
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = osp.join(save_path, data_type)  # 创建路径
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(save_path, name)

        # 创建HDF5文件，并为每个被试保存数据和对应标签
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()


    def split(self, data, label):
        """
        这个函数将一次试验的数据分成更短的片段
        Parameters
        ----------
        data: (num_segment, f, channel, data)
        label(list): num_segment

        Returns
        -------
        data: (num_segment, f, channel, data)
        label(array): (num_segment,)
        """
        data = data
        label = np.array(label)
        print("The data and label are split: Data shape:" + str(data.shape) + " Label:" + str(label.shape))
        assert len(data) == len(label)
        return data, label
