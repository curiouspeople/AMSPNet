import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import torch.nn.functional as F

from scipy.fftpack import fft
from architecture.layers import GraphConvolution, DepthwiseSeparableConv1d



# 设置GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class AMSPNet_remove_spa(nn.Module):
    def __init__(self, num_classes, input_size, batch_size, dropout_rate, idx_graph, window_size, stride, out_gcn, out_gru):
        """
        相较于Net1:
        选择权重最高一段时间的代码由注意力模块换为了挤压激励模块
        GCG中由普通卷积改为深度可分离卷积

        Parameters:
        input_size: 输入至模型的数据的维度 (b, 1, c, d)
        """
        super(AMSPNet_remove_spa, self).__init__()

        self.idx = idx_graph
        self.batch_size = batch_size                    # 批大小
        self.channel = input_size[1]                    # 通道数
        self.data_all = input_size[-1]                  # 总采样点大小

        self.brain_area = len(self.idx)                 # 脑区的数量
        self.idx_region = self.get_idx(self.idx)        # 脑区的索引

        self.window_size = window_size                  # 模型内滑动窗大小
        self.stride = stride                            # 模型内滑动窗步长
        self.segments = (self.data_all*2)/self.window_size  # 模型内滑动窗得到的段数

        self.out_gcn = out_gcn      # GCN的隐藏输出
        self.out_gru = out_gru      # GRU的隐藏输出

        self.tem = TemporalModule(self.batch_size, self.channel, self.window_size, out_gcn, out_gru, self.segments)

        self.fc_gen = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(out_gru), int(self.channel)),
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.channel), num_classes))


    def forward(self, x):   # (b, 1, c, d)
        x = x.squeeze(1)        # (b, c, d) 降维
        input = self.setFFT(x)  # (b, c, d) FFT变换

        # 滑动窗将输入的时间序列特征划分为s个50%滑动窗口的Xs时间序列段  s取决于输入的时间的大小
        # s为划分的段的数量，sd为划分的每一段对应的的数据点
        input = self.adaptive_segmentation(input, self.window_size, self.stride).permute(0, 2, 1, 3).float().to(DEVICE) # (b, s, c, sd)

        out_tem = self.tem.forward(input)                   # (b, out_gru)    时间分支信息

        tsne_out_before = out_tem                           # t-SNE输出内容:分类前
        out = self.fc_gen(out_tem)
        tsne_out_after = out                                # t-SNE输出内容:分类后

        return out, tsne_out_before, tsne_out_after


    def setFFT(self, x):
        """
        将EEG原始数据应用傅里叶变换
        Returns
        -------
        input: 应用FFT后的EEG时频信息
        """
        start_time_step = 0
        time_steps = []
        while start_time_step <= x.shape[-1] - self.window_size:
            end_time_step = start_time_step + self.window_size
            curr_time_step = x[:, :, start_time_step:end_time_step]
            curr_time_step, _ = self.computeFFT(curr_time_step, n=self.window_size)
            time_steps.append(curr_time_step)
            start_time_step = end_time_step
        input = np.stack(time_steps, axis=1)
        input = torch.from_numpy(input).permute(0, 2, 1, 3).to(DEVICE)  # (b, c, fs, fd)
        input = input.reshape(self.batch_size, self.channel, -1)        # (b, c, fs * fd)

        return input


    def computeFFT(self, x, n):
        """
        计算傅里叶变换后数据
        Returns
        -------
        FT: 信号FFT的对数幅值
        P: 信号FFT的相位谱
        """
        # 傅里叶变换
        fourier_signal = fft(x.cpu().detach().numpy(), n=n, axis=-1) # 在最后一维FFT fourier_signal:(b, c, n)

        # 只取正频部分
        idx_pos = int(np.floor(n / 2))                  # 将fourier_signal的最后一个维度截取到n/2的位置
        fourier_signal = fourier_signal[:, :idx_pos]    # 计算fourier_signal的幅度amp和相位P
        amp = np.abs(fourier_signal)                    # idx_pos:(b, c, n)
        amp[amp == 0.] = 1e-8                           # 避免log0

        FT = np.log(amp)                # FT:(b, c, n)
        P = np.angle(fourier_signal)    # P:(b, c, n)

        return FT, P


    def adaptive_segmentation(self, x, window_size, stride):
        """
        对输入的x进行自适应分段，并返回分段后的列表
        Parameters
        -------
        window_size: 每一段的长度
        stride: 滑动窗的步长

        Returns
        -------
        result: 分段后的列表 (b, s, c, sd)
        """
        batchsize, channel, data = x.shape
        divided_data = (data - window_size) // stride + 2                   # 计算分段后的段数
        result = np.zeros((batchsize, channel, divided_data, window_size))  # 初始化分段后的数据
        for i in range(batchsize):
            for j in range(channel):
                # 按行滑动窗口，取每个窗口的数据
                data_ij = x[i, j]
                segments_ij = []
                start = 0
                while start + window_size <= data:
                    segment = data_ij[start:start + window_size]
                    segments_ij.append(segment)
                    start += stride
                # 处理最后一个窗口不足的情况
                if start < data:
                    segment = torch.zeros((window_size,))
                    segment[:data - start] = data_ij[start:]
                    segment = segment.to(DEVICE)
                    segments_ij.append(segment)
                # 将分段后的数据保存下来
                segments_ij += [torch.zeros((window_size,))] * (divided_data - len(segments_ij))
                result[i, j] = torch.vstack(segments_ij).cpu().detach().numpy()
        result = torch.from_numpy(result)
        return result


    def attention(self, x1, x2):
        """
        对时间特征和空间特征应用注意力为其分配权重
        Parameters
        -------
        x1: 时间特征 (b, out_gru)
        x2: 空间特征 (b, 7*s*rc^2)

        Returns
        -------
        final_feature: 经过池化后的时空融合特征 (b, 7*s*rc^2)
        """
        out_tem = x1.size(1)
        out_spa = x2.size(1)

        out_dim = max(out_tem, out_spa)
        tem_pool = nn.AdaptiveAvgPool1d(out_dim)(x1.unsqueeze(1)).squeeze(1)
        spa_pool = nn.AdaptiveAvgPool1d(out_dim)(x2.unsqueeze(1)).squeeze(1)

        attention_layer = nn.Sequential(
            nn.Linear(out_dim, 1),  # 计算每个特征的注意力分数
            nn.Softmax(dim=1)       # 归一化为概率
        ).to(DEVICE)

        attention_weights_tem = attention_layer(tem_pool)
        attention_weights_spa = attention_layer(spa_pool)

        weighted_tem_feature = tem_pool * attention_weights_tem
        weighted_spa_feature = spa_pool * attention_weights_spa

        final_feature = weighted_tem_feature + weighted_spa_feature

        # return final_feature
        return final_feature, weighted_tem_feature, weighted_spa_feature


    def get_idx(self, chan_in_area):
        """
        根据每个区域的通道数量计算出每个区域的索引
        """
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)  # idx列表中的每个元素与前一个元素累加
        return idx_[1:]                # 返回idx_列表中从第二个元素开始的所有元素，表示每个区域的索引



### ——————————————————————————————————————————————————— 时间处理分支 ————————————————————————————————————————————————————
class TemporalModule(nn.Module):
    def __init__(self, batch_size, channel, data_seg, out_gcn, out_gru, segments):
        super().__init__()
        self.batch_size = batch_size
        self.channel = channel
        self.data_seg = data_seg

        self.attn = SqueezeAndExcitationModule(int(segments), r=2)
        self.gcg = GCG(data_seg, channel, out_gcn, out_gru)


    def forward(self, x):   # (b, s, c, sd)
        # 得到用于输入至GCG的最高权重的EEG数据
        max_seg = self.attn.forward(x)              # (b, 1, c, sd)
        adj = self.get_adj_iCOH(max_seg.squeeze(1)) # (b, c, c)
        out_tem = self.gcg.forward(x, adj)          # (b, out_gru)

        return out_tem


    def get_adj_iCOH(self, x):
        """
        计算动态功能连通邻接矩阵，此处使用的是一批数据计算iCOH
        ----------
        :return: iCOH邻接矩阵adjacency_matrix (b, c, c)
        """
        b, c, _ = x.shape

        # Fourier变换
        fft_result = torch.fft.fft(x, dim=-1)

        # 计算交叉功率谱的乘积
        cross_power_spectrum = fft_result.unsqueeze(2) * torch.conj(fft_result.unsqueeze(1))

        # 计算虚部相干性（iCOH）
        numerator = torch.mean(torch.imag(cross_power_spectrum), dim=-1)
        denominator = torch.sqrt(torch.mean(torch.abs(cross_power_spectrum), dim=-1) ** 2)

        # 设置对角线为1
        adjacency_matrix = numerator / denominator
        adjacency_matrix = adjacency_matrix * (1 - torch.eye(c, device=x.device))

        return adjacency_matrix.float()



class SqueezeAndExcitationModule(nn.Module):
    def __init__(self, s, r):
        super(SqueezeAndExcitationModule, self).__init__()

        # Squeeze操作：全局平均池化，按段数压缩
        self.squeeze = torch.nn.AdaptiveAvgPool2d(1)  # 输出形状为 (b, s, 1, 1)

        # Excitation操作：通过全连接层生成通道注意力
        self.excitation = torch.nn.Sequential(
            torch.nn.Linear(s, s // r, bias=False),  # 将段数 s 压缩到 s // r
            torch.nn.ReLU(inplace=True),  # 激活函数
            torch.nn.Linear(s // r, s, bias=False),  # 再恢复到 s
            torch.nn.Sigmoid()  # 输出注意力权重
        ).to(DEVICE)

    def forward(self, x):
        x_squeezed = self.squeeze(x)                            # (b, s, 1, 1)
        x_squeezed = x_squeezed.view(x_squeezed.size(0), -1)    # (b, s)
        attention_weights = self.excitation(x_squeezed)         # (b, s)

        attention_weights = attention_weights.view(attention_weights.size(0), attention_weights.size(1), 1, 1)  # (b, s, 1, 1)
        # 对每个段进行加权
        x_weighted = x * attention_weights

        weighted_max = x_weighted.max(dim=1, keepdim=True)[0]   # (b, 1, c, d)

        return weighted_max



class GCG(nn.Module):
    def __init__(self, data_seg, channel, out_gcn, out_gru):
        super(GCG, self).__init__()

        self.gcn = GraphConvolution(data_seg, out_gcn)
        self.out_gru = out_gru

        # GRU相关的门控操作
        self.W_r = nn.Linear(out_gcn + out_gru, out_gru)
        self.W_u = nn.Linear(out_gcn + out_gru, out_gru)
        self.W_c = nn.Linear(out_gcn + out_gru, out_gru)

        self.b_r = nn.Parameter(torch.zeros(out_gru))
        self.b_u = nn.Parameter(torch.zeros(out_gru))
        self.b_c = nn.Parameter(torch.zeros(out_gru))

        self.conv1 = DepthwiseSeparableConv1d(channel, channel, 64)
        self.conv2 = DepthwiseSeparableConv1d(channel, channel, 32)
        self.conv3 = DepthwiseSeparableConv1d(channel, channel, 14)


    def forward(self, x, adj):
        """
        这个函数实现GCN和GRU的结合，以对每一秒的数据提取其时空特征，返回最后一个时刻点的时间数据

        Returns:
        ----------
        h_t_last: 最后一个时刻点的时间数据 (b, out_gru)
        """
        b, s, c, _ = x.size()

        # 初始化GRU的隐藏状态
        h_t = torch.zeros(b, self.out_gru).to(DEVICE)   # (b, out_gru) GRU的隐藏状态

        # 遍历每个时刻
        for t in range(s):
            curr_x = x[:, t, :, :]                  # (b, c, sd) 当前时刻的数据

            # 卷积学习该段时间特征以更新邻接矩阵
            adj_delta = self.conv1(curr_x)
            adj_delta = self.conv2(adj_delta)
            adj_delta = self.conv3(adj_delta)
            adj_t = adj_delta + adj

            # 使用 sigmoid 限制邻接矩阵的值域
            adj_t = torch.sigmoid(adj_t)            # (b, c, c)

            # 图卷积处理当前时刻点的数据
            gcn_out = self.gcn(curr_x, adj_t)       # (b, c, out_gcn)
            gcn_out_flat = gcn_out.view(b * c, -1)  # (b * c, out_gcn)

            if t == 0:
                h_t_flat = torch.zeros(b * c, self.out_gru).to(DEVICE)  # (b * c, out_gru) 初始化为零向量H^0
            else:
                h_t_flat = h_t.view(b * c, -1)                          # 其他时刻使用上一时刻的H^t-1

            # 计算门控
            r_t = torch.sigmoid(self.W_r(torch.cat([gcn_out_flat, h_t_flat], dim=-1)) + self.b_r)       # (b * c, out_gru)
            u_t = torch.sigmoid(self.W_u(torch.cat([gcn_out_flat, h_t_flat], dim=-1)) + self.b_u)       # (b * c, out_gru)

            # 计算候选隐藏状态 c_t
            c_t = torch.tanh(self.W_c(torch.cat([gcn_out_flat, r_t * h_t_flat], dim=-1)) + self.b_c)    # (b * c, out_gru)

            # 更新隐藏状态 h_t
            h_t_flat = u_t * h_t_flat + (1 - u_t) * c_t # (b * c, out_gru)
            h_t = h_t_flat.view(b, c, -1)               # (b, c, out_gru)

        # 取最后时刻的隐藏状态
        h_t_last = h_t[:, -1, :]  # (b, out_gru)

        return h_t_last

### ——————————————————————————————————————————————————— 时间处理分支 ————————————————————————————————————————————————————
