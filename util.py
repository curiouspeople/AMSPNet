import os
import time
import pprint
import random

from architecture.AMSPNet import *
from architecture.AMSPNet_remove_tem import *
from architecture.AMSPNet_remove_spa import *
from architecture.AMSPNet_Sim import *
from architecture.AMSPNet_fixed_iCOH import *
from architecture.AMSPNet_GCL import *
from architecture.AMSPNet_band import *

from dataset.eeg_dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score


def set_gpu(x):
    """
    设置使用gpu运行模型
    """
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def seed_all(seed):
    """
    设置使用随机种子
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def seed_print():
    state = random.getstate()
    print("—————————————————————————————————————————————————————————————————————————————————————————————————————————————")
    print("当前随机种子为:", state[1][0])
    print("—————————————————————————————————————————————————————————————————————————————————————————————————————————————")


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


class Averager():
    """
    计算平均值
    """
    def __init__(self):
        self.n = 0  # 当前添加的数字的数量
        self.v = 0  # 存储当前的平均值

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    """
    计算分类模型的准确率
    """
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def get_model(args):
    """
    用于获取模型参数并创建模型
    """
    if args.model == 'AMSPNet':
        idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
        channels = sum(idx_local_graph)
        input_size = (args.input_shape[0], channels, args.input_shape[2])
        model = AMSPNet(
            num_classes=args.num_class, input_size=input_size, batch_size=args.batch_size,
            dropout_rate=args.dropout, idx_graph=idx_local_graph,
            window_size=args.window_size, stride=args.stride,
            out_gcn=args.out_gcn, out_gru=args.out_gru
        )

    elif args.model == 'AMSPNet_band':
        idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
        channels = sum(idx_local_graph)
        input_size = (args.input_shape[0], channels, args.input_shape[2])
        model = AMSPNet_band(
            num_classes=args.num_class, input_size=input_size, batch_size=args.batch_size,
            dropout_rate=args.dropout, idx_graph=idx_local_graph,
            window_size=args.window_size, stride=args.stride,
            out_gcn=args.out_gcn, out_gru=args.out_gru
        )

    elif args.model == 'AMSPNet_remove_tem':
        idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
        channels = sum(idx_local_graph)
        input_size = (args.input_shape[0], channels, args.input_shape[2])
        model = AMSPNet_remove_tem(
            num_classes=args.num_class, input_size=input_size, batch_size=args.batch_size,
            dropout_rate=args.dropout, idx_graph=idx_local_graph,
            window_size=args.window_size, stride=args.stride,
            out_gcn=args.out_gcn, out_gru=args.out_gru
        )

    elif args.model == 'AMSPNet_remove_spa':
        idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
        channels = sum(idx_local_graph)
        input_size = (args.input_shape[0], channels, args.input_shape[2])
        model = AMSPNet_remove_spa(
            num_classes=args.num_class, input_size=input_size, batch_size=args.batch_size,
            dropout_rate=args.dropout, idx_graph=idx_local_graph,
            window_size=args.window_size, stride=args.stride,
            out_gcn=args.out_gcn, out_gru=args.out_gru
        )

    elif args.model == 'AMSPNet_fixed_iCOH':
        idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
        channels = sum(idx_local_graph)
        input_size = (args.input_shape[0], channels, args.input_shape[2])
        model = AMSPNet_fixed_iCOH(
            num_classes=args.num_class, input_size=input_size, batch_size=args.batch_size,
            dropout_rate=args.dropout, idx_graph=idx_local_graph,
            window_size=args.window_size, stride=args.stride,
            out_gcn=args.out_gcn, out_gru=args.out_gru
        )

    elif args.model == 'AMSPNet_GCL':
        idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
        channels = sum(idx_local_graph)
        input_size = (args.input_shape[0], channels, args.input_shape[2])
        model = AMSPNet_GCL(
            num_classes=args.num_class, input_size=input_size, batch_size=args.batch_size,
            dropout_rate=args.dropout, idx_graph=idx_local_graph,
            window_size=args.window_size, stride=args.stride,
            out_gcn=args.out_gcn, out_gru=args.out_gru
        )

    elif args.model == 'AMSPNet_Sim':
        idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
        channels = sum(idx_local_graph)
        input_size = (args.input_shape[0], channels, args.input_shape[2])
        model = AMSPNet_Sim(
            num_classes=args.num_class, input_size=input_size, batch_size=args.batch_size,
            dropout_rate=args.dropout, idx_graph=idx_local_graph,
            window_size=args.window_size, stride=args.stride,
            out_gcn=args.out_gcn, out_gru=args.out_gru
        )
    return model



def get_dataloader(data, label, batch_size):
    """
    获取dataloader对象
    """

    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    return loader


def get_metrics(y_pred, y_true, classes=None):
    """
    获取最后输出的指标
    :param y_pred: 模型预测的标签
    :param y_true: 实际的标签
    :param classes: 类别列表
    :return: 准确率，F1分数，混淆矩阵
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return acc, sensitivity, specificity, f1, kappa, cm


def get_metrics_individual(y_pred, y_true):
    """
    获取最后输出的指标
    :param y_pred: 模型预测的标签
    :param y_true: 实际的标签
    :param classes: 类别列表
    :return: 准确率，F1分数，混淆矩阵
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, f1, kappa


def get_trainable_parameter_num(model):
    """
    计算模型中可训练参数的数量
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    # 来遍历模型的参数，并筛选出那些requires_grad属性为True的参数。
    return total_params


def L1Loss(model, Lambda):
    """
    L1损失函数
    :param model: 模型对象
    :param lambda: 正则化参数
    :return: 函数的返回值
    """
    w = torch.cat([x.view(-1) for x in model.parameters()]) # 遍历模型参数，并将参数展平为一维向量，再使用torch.cat将所有参数的一维向量连接起来
    err = Lambda * torch.sum(torch.abs(w))
    return err


def L2Loss(model, Lambda):
    """
    L2损失函数
    :param model: 模型对象
    :param lambda: 正则化参数
    :return: 函数的返回值
    """
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(w.pow(2))
    return err


class LabelSmoothing(nn.Module):
    """
    带标签平滑的NLL损耗
    参考: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """
    def __init__(self, smoothing=0.0):
        """LabelSmoothing模块的构造函数
        :param smoothing: 标签平滑系数
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing   # 置信度
        self.smoothing = smoothing  # 平滑系数

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)   # 得到对数概率
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))  # 得到负对数似然损失
        nll_loss = nll_loss.squeeze(1)      # 去除维度为1的尺寸，将负对数似然损失变为一维张量
        smooth_loss = -logprobs.mean(dim=-1)    # 计算最终的损失
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss    # 计算最终的损失
        return loss.mean()
