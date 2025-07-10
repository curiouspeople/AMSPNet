from n_fold_cross_validation import *
from prepare_data import *
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ######## Record ########
    parser.add_argument('--record', type=bool, default=False, choices=[True, False])

    ######## Data ########
    parser.add_argument('--dataset', type=str, default='EEG_1_10min')
    parser.add_argument('--data-path', type=str, default='dataset/EEG_1_10min')
    parser.add_argument('--subjects', type=int, default=27, choices=[27, 34, 61])
    parser.add_argument('--num-class', type=int, default=2)
    parser.add_argument('--label-type', type=str, default='float')
    parser.add_argument('--segment', type=int, default=10)                      # 5 输入模型的EEG时间长度
    parser.add_argument('--overlap', type=float, default=0.5)                   # 0.5 用于EEG数据增强的滑动窗大小
    parser.add_argument('--sampling-rate', type=int, default=128)
    parser.add_argument('--scale-coefficient', type=float, default=1)
    parser.add_argument('--input-shape', type=tuple, default=(1, 32, 128 * 10))  # sampling-rate * segment
    parser.add_argument('--data-format', type=str, default='eeg')

    ######## Training Process ########
    parser.add_argument('--random-seed', type=int, default=2024)        # 2024
    parser.add_argument('--max-epoch', type=int, default=100)           # 100
    parser.add_argument('--patient', type=int, default=30)              # 30
    parser.add_argument('--batch-size', type=int, default=128)          # 256
    parser.add_argument('--learning-rate', type=float, default=25e-4)   # 25e-4
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)           # 0.5
    parser.add_argument('--L2-rate', type=float, default=0)
    parser.add_argument('--LS', type=bool, default=True, help="Label smoothing")
    parser.add_argument('--LS-rate', type=float, default=0.1)           # 0.1

    parser.add_argument('--train-rate', type=float, default=0.9)        # 0.9
    parser.add_argument('--save-path', default='save/cross')
    parser.add_argument('--load-path-final', default='save/cross/final_model.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-model', type=bool, default=True)

    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='AMSPNet')
    parser.add_argument('--graph-type', type=str, default='gen', choices=['gen'])           # 脑区划分类型
    parser.add_argument('--window-size', type=int, default=128)                             # 128 dFC计算的滑动窗大小
    parser.add_argument('--stride', type=int, default=64)                                   # 64  dFC计算的滑动窗步长

    parser.add_argument('--out-gcn', type=int, default=16)                                  # GCN的隐藏输出
    parser.add_argument('--out-gru', type=int, default=16)                                  # GRU的隐藏输出

    ######## Reproduce the result using the saved model ######
    parser.add_argument('--reproduce', action='store_true')

    ######## Change last time ######
    parser.add_argument('--change', type=str, default='5折交叉验证; 数据集：EEG_1_10min; model：AMSPNet;')

    # 参数初始化过程
    args = parser.parse_args()                      # 初始化所有参数

    # 被试数据初始化过程
    sub_to_run = np.arange(args.subjects)           # 创建列表存储被试数量
    pd = PrepareData(args)                          # 读取被试数据参数
    pd.run(sub_to_run, split=True, expand=True)     # 被试数据初始化

    # n折交叉验证模型训练过程
    cv = NFoldCrossValidation(args)               # 初始化交叉验证参数
    seed_all(args.random_seed)                    # 随机种子初始化

    cv.n_fold_CV(subject=sub_to_run)              # 进行n折交叉验证