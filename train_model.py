import numpy as np
import torch

from util import *
from toolbox import *
from experiment.tSNE import *
from experiment.Topmap import *
from experiment.HeatMap import *



CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tsne_root = './recording/tSNE'  # 此处注意服务器与本地区别
visual_root = './recording/visual'  # 此处注意服务器与本地区别





# 训练模式
def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()     # 设置为训练模式
    tl = Averager() # 计算平均损失
    pred_train = []
    act_train = []

    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
        out, tsne_out_before, tsne_out_after, adj = net(x_batch)
        loss = loss_fn(out, y_batch)

        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())

    return tl.item(), pred_train, act_train, tsne_out_before, tsne_out_after, y_batch



# 验证模式
def valid(data_loader, net, loss_fn):
    net.eval()      # 设置为验证模式
    vl = Averager() # 计算平均损失
    pred_val = []
    act_val = []

    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
            out, tsne_out_before, tsne_out_after, adj = net(x_batch)
            loss = loss_fn(out, y_batch)

            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())

    return vl.item(), pred_val, act_val, tsne_out_before, tsne_out_after, y_batch



# 测试模式
def predict(data_loader, net, loss_fn):
    net.eval()      # 设置为验证模式
    vl = Averager() # 计算平均损失
    pred_val = []
    act_val = []
    out_val = None
    tsne_before_out_all = None
    tsne_after_out_all = None
    y_batch_all = None

    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
            out, tsne_out_before, tsne_out_after, adj = net(x_batch)
            loss = loss_fn(out, y_batch)

            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
            if out_val is None:
                out_val = out
            else:
                out_val = torch.cat((out_val, out), dim=0)


            if tsne_before_out_all is None:
                tsne_before_out_all = tsne_out_before
            else:
                tsne_before_out_all = torch.cat((tsne_before_out_all, tsne_out_before), dim=0)

            if tsne_after_out_all is None:
                tsne_after_out_all = tsne_out_after
            else:
                tsne_after_out_all = torch.cat((tsne_after_out_all, tsne_out_after), dim=0)

            if y_batch_all is None:
                y_batch_all = y_batch
            else:
                y_batch_all = torch.cat((y_batch_all, y_batch), dim=0)

    return vl.item(), pred_val, act_val, tsne_before_out_all, tsne_after_out_all, y_batch_all, adj
    # return vl.item(), pred_val, act_val, tsne_before_out_all, tsne_after_out_all, y_batch_all, adj, out_val[:,1].cpu().squeeze().numpy()



def set_up(args):
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True



def train(args, dir, data_train, label_train, data_val, label_val, out_fold):
    """
    此函数用于获取验证集的最大acc和f1-score
    :return: 验证集最大acc及其对应f1-score
    """
    TL = []
    TA = []
    VL = []
    VA = []

    # set_acc = 0.9900
    tsne_before_out_train_all = None
    tsne_after_out_train_all = None
    tsne_label_train_all = None

    tsne_before_out_val_all = None
    tsne_after_out_val_all = None
    tsne_label_val_all = None

    label_names = ['MCI', 'HC']
    seed_all(args.random_seed)
    save_name = 'fold' + str(out_fold)
    set_up(args)

    # 初始化训练集和测试集的dataloader对象
    train_loader = get_dataloader(data_train, label_train, args.batch_size)
    val_loader = get_dataloader(data_val, label_val, args.batch_size)

    # 初始化模型
    model = get_model(args)
    if CUDA:
        model = model.to(DEVICE)

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # 初始化损失函数
    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate).to(DEVICE)
    else:
        loss_fn = nn.CrossEntropyLoss().to(DEVICE)


    def save_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), osp.join(args.save_path, '{}.pth'.format(name)))


    def save_whole_model(name):
        previous_model = osp.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model, osp.join(args.save_path, '{}.pth'.format(name)))


    # 创建trlog字典对象保存参数和指标
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['F1'] = 0.0
    trlog['total_params'] = 0.0

    timer = Timer()
    patient = args.patient
    counter = 0

    for epoch in range(1, args.max_epoch + 1):
        loss_per_train = []
        acc_per_train = []
        loss_per_val = []
        acc_per_val = []

        # 计算训练集损失，预测值和实际值
        loss_train, pred_train, act_train, tsne_before_out_train, tsne_after_out_train, tsne_label_train = train_one_epoch(data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)

        # 计算训练集acc和f1-score
        acc_train, _, _, f1_train, _, _ = get_metrics(y_pred=pred_train, y_true=act_train)
        print('Stage 1 : epoch {}, train, loss={:.4f} acc={:.4f} f1={:.4f}'.format(epoch, loss_train, acc_train, f1_train))

        loss_per_train.append(loss_train)
        acc_per_train.append(acc_train)
        loss_per_train = np.array(loss_per_train)
        acc_per_train = np.array(acc_per_train)


        if tsne_before_out_train_all is None:
            tsne_before_out_train_all = tsne_before_out_train
        else:
            tsne_before_out_train_all = torch.cat((tsne_before_out_train_all, tsne_before_out_train), dim=0)

        if tsne_after_out_train_all is None:
            tsne_after_out_train_all = tsne_after_out_train
        else:
            tsne_after_out_train_all = torch.cat((tsne_after_out_train_all, tsne_after_out_train), dim=0)

        if tsne_label_train_all is None:
            tsne_label_train_all = tsne_label_train
        else:
            tsne_label_train_all = torch.cat((tsne_label_train_all, tsne_label_train), dim=0)


        # 计算验证集损失，预测值和实际值
        loss_val, pred_val, act_val, tsne_before_out_val, tsne_after_out_val, tsne_label_val = valid(data_loader=val_loader, net=model, loss_fn=loss_fn)
        # 计算验证集acc和f1-score
        acc_val, _, _, f1_val, _, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('Stage 1 : epoch {}, valid, loss={:.4f} acc={:.4f} f1={:.4f}'.format(epoch, loss_val, acc_val, f1_val))

        loss_per_val.append(loss_val)
        acc_per_val.append(acc_val)
        loss_per_val = np.array(loss_per_val)
        acc_per_val = np.array(acc_per_val)


        if tsne_before_out_val_all is None:
            tsne_before_out_val_all = tsne_before_out_val
        else:
            tsne_before_out_val_all = torch.cat((tsne_before_out_val_all, tsne_before_out_val), dim=0)

        if tsne_after_out_val_all is None:
            tsne_after_out_val_all = tsne_after_out_val
        else:
            tsne_after_out_val_all = torch.cat((tsne_after_out_val_all, tsne_after_out_val), dim=0)

        if tsne_label_val_all is None:
            tsne_label_val_all = tsne_label_val
        else:
            tsne_label_val_all = torch.cat((tsne_label_val_all, tsne_label_val), dim=0)


        # 获取验证集最高acc
        if acc_val >= trlog['max_acc']:
            trlog['max_acc'] = acc_val
            trlog['F1'] = f1_val
            save_model('candidate')
            save_whole_model('whole_final_model_fold{}'.format(out_fold+1))
            counter = 0
        else:
            counter += 1
            if counter >= patient:
                print('early stopping')

                if args.record:
                    title = "Progress_of_train_data_before"
                    tSNE_2D_nfold(tsne_root, tsne_before_out_train_all.cpu().detach().numpy(), tsne_label_train_all, title, out_fold, label_names)
                    title = "Progress_of_train_data_after"
                    tSNE_2D_nfold(tsne_root, tsne_after_out_train_all.cpu().detach().numpy(), tsne_label_train_all, title, out_fold, label_names)

                    title = "Progress_of_val_data_before"
                    tSNE_2D_nfold(tsne_root, tsne_before_out_val_all.cpu().detach().numpy(), tsne_label_val_all, title, out_fold, label_names)
                    title = "Progress_of_val_data_after"
                    tSNE_2D_nfold(tsne_root, tsne_after_out_val_all.cpu().detach().numpy(), tsne_label_val_all, title, out_fold, label_names)

                break

        TL.append(np.mean(loss_per_train))
        TA.append(np.mean(acc_per_train))
        VL.append(np.mean(loss_per_val))
        VA.append(np.mean(acc_per_val))

        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)
        print('ETA:{}/{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch), out_fold+1))

    # 保存训练日志文件
    save_name = 'trlog' + save_name
    experiment_setting = 'W_{}_GC_{}_GR_{}'.format(args.window_size, args.out_gcn, args.out_gru)
    save_path = osp.join(args.save_path, experiment_setting, 'log_train')
    ensure_path(save_path)
    torch.save(trlog, osp.join(save_path, save_name))

    trlog['total_params'] = sum(p.numel() for p in model.parameters())

    if args.record:
        # 绘制该fold的训练集和测试集的准确率图和损失图
        pltLoss_TV_nfold(dir, out_fold, TL, VL)
        pltAcc_TV_nfold(dir, out_fold, TA, VA)

        title = "Progress_of_train_data_before"
        tSNE_2D_nfold(tsne_root, tsne_before_out_train_all.cpu().detach().numpy(), tsne_label_train_all, title, out_fold, label_names)
        title = "Progress_of_train_data_after"
        tSNE_2D_nfold(tsne_root, tsne_after_out_train_all.cpu().detach().numpy(), tsne_label_train_all, title, out_fold, label_names)

        title = "Progress_of_val_data_before"
        tSNE_2D_nfold(tsne_root, tsne_before_out_val_all.cpu().detach().numpy(), tsne_label_val_all, title, out_fold, label_names)
        title = "Progress_of_val_data_after"
        tSNE_2D_nfold(tsne_root, tsne_after_out_val_all.cpu().detach().numpy(), tsne_label_val_all, title, out_fold, label_names)

    return trlog['max_acc'], trlog['F1'], trlog['total_params']



def test(args, data, label, reproduce, fold):
    """
    此函数用于获取测试集的acc和预测值和实际值
    :return: 测试集的acc和预测值和实际值
    """
    set_up(args)
    seed_all(args.random_seed)
    # 初始化训练集dataloader对象
    test_loader = get_dataloader(data, label, args.batch_size)

    # 初始化模型
    model = get_model(args)

    if CUDA:
        model = model.to(DEVICE)

    # 初始化损失函数
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    # ————————————————————————————————————————————————————此处用于复现模型————————————————————————————————————————————————
    if reproduce:
        model_name_reproduce = 'fold' + str(fold) + '.pth'
        data_type = 'model_{}_{}'.format(args.data_format, args.label_type)
        experiment_setting = 'W_{}_GC_{}_GR_{}'.format(args.window_size, args.out_gcn, args.out_gru)
        load_path_final = osp.join(args.save_path, experiment_setting, data_type, model_name_reproduce)
        model.load_state_dict(torch.load(load_path_final))
    # ————————————————————————————————————————————————————此处用于复现模型————————————————————————————————————————————————

    else:
        model.load_state_dict(torch.load(args.load_path_final))

    # 计算测试集的损失，预测值和实际值
    # loss, pred, act, tsne_out_before_test, tsne_out_after_test, tsne_label_test, adj, out = predict(data_loader=test_loader, net=model, loss_fn=loss_fn)
    loss, pred, act, tsne_out_before_test, tsne_out_after_test, tsne_label_test, adj = predict(data_loader=test_loader, net=model, loss_fn=loss_fn)

    # 测试集准确率 f1-score 混淆矩阵
    acc, sen, spe, f1, kappa, cm = get_metrics(y_pred=pred, y_true=act)

    if args.record:
        label_names = ['MCI', 'HC']
        title = "Progress_of_test_data_before"
        tSNE_2D_nfold(tsne_root, tsne_out_before_test.cpu().detach().numpy(), tsne_label_test, title, fold, label_names)
        title = "Progress_of_test_data_after"
        tSNE_2D_nfold(tsne_root, tsne_out_after_test.cpu().detach().numpy(), tsne_label_test, title, fold, label_names)

    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
    return acc, pred, act