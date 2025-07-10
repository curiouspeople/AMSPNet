import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class GraphConvolution(Module):
    """
    简单GCN层
    Parameters:
    in_features: 输入特征维度
    out_features：输出特征
    bias：是否使用偏执项
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))   # 可学习参数weight，形状为(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)                  # 对参数进行初始化

        if bias:
            # 创建一个名为bias的可学习参数，其形状为(1, 1, out_features)，使用torch.zeros方法初始化参数
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)


    # x：输入特征 adj：邻接矩阵
    def forward(self, x, adj):
        output = torch.matmul(x, self.weight) - self.bias   # 输入特征与权重相乘，再减去偏执项
        output = F.relu(torch.matmul(adj, output))
        return output



class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeparableConv1d, self).__init__()
        # 深度卷积层：每个输入通道有自己的卷积核
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, groups=in_channels)
        # 逐点卷积层：使用1x1卷积来改变通道数
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # 先通过深度卷积，再通过逐点卷积
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
