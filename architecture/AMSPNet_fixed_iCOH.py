import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import torch.nn.functional as F

from scipy.fftpack import fft
from architecture.layers import GraphConvolution, DepthwiseSeparableConv1d



# 设置GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class AMSPNet_fixed_iCOH(nn.Module):
    def __init__(self, num_classes, input_size, batch_size, dropout_rate, idx_graph, window_size, stride, out_gcn, out_gru):
        """
        邻接矩阵固定iCOH计算得到

        Parameters:
        input_size: 输入至模型的数据的维度 (b, 1, c, d)
        """
        super(AMSPNet_fixed_iCOH, self).__init__()

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
        self.spa = SpatialModule(self.idx)

        self.fc_gen = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int((self.segments)*7*3*3), int(self.channel)),
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.channel), num_classes))


    def forward(self, x):   # (b, 1, c, d)
        x = x.squeeze(1)        # (b, c, d) 降维
        input = self.setFFT(x)  # (b, c, d) FFT变换

        # 滑动窗将输入的时间序列特征划分为s个50%滑动窗口的Xs时间序列段  s取决于输入的时间的大小
        # s为划分的段的数量，sd为划分的每一段对应的的数据点
        input = self.adaptive_segmentation(input, self.window_size, self.stride).permute(0, 2, 1, 3).float().to(DEVICE) # (b, s, c, sd)

        out_tem = self.tem.forward(input)                   # (b, out_gru)    时间分支信息
        out_spa = self.spa.forward(input)                   # (b, 7*rc, s*rc) 空间分支信息
        out_spa = out_spa.view(self.batch_size, -1)         # (b, 7*s*rc^2)

        out_confusion, weighted_tem, weighted_spa = self.attention(out_tem, out_spa)    # (b, 7*s*rc^2)/(b, out_gru)

        tsne_out_before = out_confusion                     # t-SNE输出内容:分类前
        out = self.fc_gen(out_confusion)
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

        self.gcg = GCG(data_seg, channel, out_gcn, out_gru)


    def forward(self, x):   # (b, s, c, sd)
        out_tem = self.gcg.forward(x)          # (b, out_gru)
        return out_tem



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


    def forward(self, x):
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

            adj_t = self.get_adj_iCOH(curr_x)       # (b, c, c) 计算固定iCOH邻接矩阵
            adj_t = torch.sigmoid(adj_t)            # (b, c, c) 使用 sigmoid 限制邻接矩阵的值域

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

### ——————————————————————————————————————————————————— 时间处理分支 ————————————————————————————————————————————————————



### ——————————————————————————————————————————————————— 空间处理分支 ————————————————————————————————————————————————————
class SpatialModule(nn.Module):
    def __init__(self, idx_area):
        super().__init__()
        self.idx = self.get_idx(idx_area)   # 脑区索引
        self.area = len(idx_area)           # 脑区数量

        # 设定脑区组合的排列顺序 LF:0, RF:1, LT:2, C:3, RT:4, P:5, O:6
        self.brain_area_combinations = [
            (0, 1, 2),  # (LF, RF, LT)
            (1, 0, 4),  # (RF, LF, RT)
            (2, 0, 5),  # (LT, LF, P)
            (3, 2, 4),  # (C,  LT, RT)
            (4, 1, 6),  # (RT, RF, O)
            (5, 3, 6),  # (P,  C,  O)
            (6, 3, 5)   # (O,  C,  P)
        ]

        self.mbfca = MBFCA()


    def forward(self, x):   # (b, s, c, sd)
        b, s, _, sd = x.size()
        out_SAB_all = []

        # 得到s个时刻点的EEG数据
        for t in range(s):
            curr_x = x[:, t, :, :]                              # (b, c, sd) 每个时刻点的数据
            area_data_all = torch.zeros(self.area, b, 3, sd)    # (7, b, rc, sd)

            # 得到7个脑区数据
            for i in range(self.area):
                if i < self.area - 1:
                    # 获取当前脑区的通道数据
                    area_data = curr_x[:, self.idx[i]:self.idx[i + 1], :]
                else:
                    # 获取最后一个脑区的通道数据
                    area_data = curr_x[:, self.idx[i]:, :]
                area_data_all[i] = area_data        # (7, b, rc, sd)

            out_SAB = self.SAB(area_data_all)       # (b, 7*rc, rc)
            out_SAB_all.append(out_SAB)
        out_SAB_all = torch.cat(out_SAB_all, dim=2) # (b, 7*rc, s*rc)

        return out_SAB_all


    def SAB(self, x):   # (7, b, rc, sd)
        outputs = []

        # 遍历每个脑区组合
        for combination in self.brain_area_combinations:
            iCOH_matrices = []

            # 获取当前组合的3个脑区索引
            i, j, k = combination
            brain_area_data = torch.stack([x[i], x[j], x[k]], dim=0)    # (3, b, rc, sd) 此脑区对应的输入MBFCA的数据

            for i in range(3):
                brain_data = brain_area_data[i]                 # (b, c, d) 获取第i个脑区的数据
                icoh_matrix = self.get_adj_iCOH(brain_data)     # (b, c, c) 获取该脑区的iCOH邻接矩阵
                iCOH_matrices.append(icoh_matrix)
            iCOH_matrices = torch.stack(iCOH_matrices, dim=0)   # (3, b, c, c) 3个脑区的邻接矩阵拼接

            # 将数据输入到 MBFCA 模块进行处理
            output = self.mbfca.forward(iCOH_matrices)
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs


    def get_idx(self, chan_in_area):
        """
        根据每个区域的通道数量计算出每个区域的索引
        """
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)  # idx列表中的每个元素与前一个元素累加
        return idx_[1:]  # 返回idx_列表中从第二个元素开始的所有元素，表示每个区域的索引


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



class MBFCA(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义每个脑区对应的 Q, K, V 变换矩阵
        self.W_Q_a = nn.Linear(3, 3)
        self.W_K_a = nn.Linear(3, 3)
        self.W_V_a = nn.Linear(3, 3)

        self.W_Q_b = nn.Linear(3, 3)
        self.W_K_b = nn.Linear(3, 3)
        self.W_V_b = nn.Linear(3, 3)

        self.W_Q_c = nn.Linear(3, 3)
        self.W_K_c = nn.Linear(3, 3)
        self.W_V_c = nn.Linear(3, 3)

        self.BN = nn.BatchNorm1d(3 * 3) # 对拼接后的3*3进行归一化
        self.fc = nn.Linear(3 * 3, 3)   # 全连接层，将维度降到3


    def forward(self, x):   # (3, b, rc, rc)
        _, b, _, _ = x.size()
        x = x.permute(1, 0, 2, 3).to(DEVICE)    # (b, 3, rc, rc)

        # 对每个脑区生成 Q, K, V
        Q_a = self.W_Q_a(x[:, 0, :, :])  # (b, rc, rc)
        K_a = self.W_K_a(x[:, 0, :, :])
        V_a = self.W_V_a(x[:, 0, :, :])

        Q_b = self.W_Q_b(x[:, 1, :, :])  # (b, rc, rc)
        K_b = self.W_K_b(x[:, 1, :, :])
        V_b = self.W_V_b(x[:, 1, :, :])

        Q_c = self.W_Q_c(x[:, 2, :, :])  # (b, rc, rc)
        K_c = self.W_K_c(x[:, 2, :, :])
        V_c = self.W_V_c(x[:, 2, :, :])

        # 注意力计算公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
        attention_1 = self.scaled_dot_product_attention(Q_c, K_a, V_b)  # (b, rc, rc)
        attention_2 = self.scaled_dot_product_attention(Q_a, K_b, V_c)  # (b, rc, rc)
        attention_3 = self.scaled_dot_product_attention(Q_b, K_c, V_a)  # (b, rc, rc)
        attention_output = torch.cat((attention_1, attention_2, attention_3), dim=-1)  # (b, rc, 3*rc)

        attention_output = self.BN(attention_output.view(-1, 3 * 3))
        attention_output = self.fc(attention_output)
        attention_output = attention_output.view(-1, 3, 3)

        return attention_output


    def scaled_dot_product_attention(self, Q, K, V):
        """
        计算缩放点积注意力
        Q, K, V: shape=(b, d)
        返回: shape=(b, d)
        """
        d_k = Q.size(-1)  # Q的最后一维大小，通常是头的维度
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # (b, d) * (b, d) -> (b, b)
        attn_weights = F.softmax(scores, dim=-1)  # (b, b)
        output = torch.matmul(attn_weights, V)  # (b, b) * (b, d) -> (b, d)
        return output
### ——————————————————————————————————————————————————— 空间处理分支 ————————————————————————————————————————————————————
