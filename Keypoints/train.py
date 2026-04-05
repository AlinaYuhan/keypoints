import random
import argparse
import contextlib
import torch
import torch.optim as optim
import numpy as np
import os  # 新增：用于系统路径检索
from merger.data_flower import all_h5
#from merger.merger_net1 import Net
from merger.merger_net2 import Net1
from merger.merger_net import Net
from merger.composed_chamfer import composed_sqrt_chamfer#calc_cd,overlap_loss_torch_error
#from torch.utils.tensorboard import SummaryWriter

arg_parser = argparse.ArgumentParser(description="Training Skeleton Merger. Valid .h5 files must contain a 'data' array of shape (N, n, 3) and a 'label' array of shape (N, 1).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
arg_parser.add_argument('-t', '--train-data-dir', type=str, default=r'E:\Education\Research\keypoints\Keypoints\point\train\h5',
                        help='Directory that contains training .h5 files.')
arg_parser.add_argument('-v', '--val-data-dir', type=str, default=r'E:\Education\Research\keypoints\Keypoints\point\train\h5',
                        help='Directory that contains validation .h5 files.')
arg_parser.add_argument('-c', '--subclass', type=int, default=14,
                        help='Subclass label ID to train on.')  # 14 is `chair` class. 27 is `guitar` class. 18 is 'table'.
arg_parser.add_argument('-m', '--checkpoint-path', '--model-path', type=str, default='chair_nolc.pt',
                        help='Model checkpoint file path for saving.')
arg_parser.add_argument('-k', '--n-keypoint', type=int, default=10,
                        help='Requested number of keypoints to detect.')
arg_parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Pytorch device for training.')
arg_parser.add_argument('-b', '--batch', type=int, default=8,
                        help='Batch size.')
arg_parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs to train.')
arg_parser.add_argument('--max-points', type=int, default=2048,
                        help='Indicates maximum points in each input point cloud.')


def L2(embed):
    return 0.01 * (torch.sum(embed ** 2))


def feed(net, optimizer, x_set, train, shuffle, batch, epoch):
    #writer = SummaryWriter("logs")
    running_loss = 0.0
    running_lrc = 0.0
    running_ldiv = 0.0
    net.train(train)
    if shuffle:
        x_set = list(x_set)
        random.shuffle(x_set)
    with contextlib.suppress() if train else torch.no_grad():
        for  i in range(len(x_set) // batch):

            idx = slice(i * batch, (i + 1) * batch)
            refp = next(net.parameters())
            batch_x = torch.tensor(np.array(x_set[idx]), device='cuda')
            #print(batch_x.device)
            if train:
                optimizer.zero_grad()
            RPCD,KPCD, LF, MA = net(batch_x)
            blrc = composed_sqrt_chamfer(batch_x, RPCD, MA)
            #blrc = calc_cd(batch_x, RPCD)
            bldiv = L2(LF)
            #bldiv = overlap_loss_torch_error(KPCD)
            loss = blrc + bldiv
            if train:
                loss.backward()
                optimizer.step()
    
            # print statistics
            running_lrc += blrc.item()
            #running_ldiv += bldiv.item()
            running_loss += loss.item()

            #writer.add_scalar('loss', loss.item(), epoch)
            print('[%s%d, %4d] loss: %.4f Lrc: %.4f Ldiv: %.4f' %
                   ('VT'[train], epoch, i, running_loss / (i + 1), running_lrc / (i + 1), running_ldiv / (i + 1)))
    return running_loss / (i + 1), running_lrc / (i + 1), running_ldiv / (i + 1)


if __name__ == '__main__':
    ns = arg_parser.parse_args()
    DATASET = ns.train_data_dir
    TESTSET = ns.val_data_dir
    batch = ns.batch
    print(DATASET)
    x, xl = all_h5(DATASET, True, True, subclasses=(ns.subclass,), sample=None)  # n x 2048 x 3
    
    x_test, xl_test = all_h5(TESTSET, True, True, subclasses=(ns.subclass,), sample=None)
    net = Net1(2048,10).to(ns.device)
    #net = Net(2048,32).to(ns.device)
    optimizer = optim.Adadelta(net.parameters(), eps=1e-2)

    # ==================== 新增：模型断点加载逻辑 (Resume Training) ====================
    start_epoch = 0
    if os.path.exists(ns.checkpoint_path):
        print(f"正在从现有检查点加载模型参数: {ns.checkpoint_path} ...")
        checkpoint = torch.load(ns.checkpoint_path, map_location=ns.device)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"加载成功。模型将从第 {start_epoch} 轮（Epoch）继续迭代。")
    else:
        print("未检测到预训练检查点，将从随机初始化状态开始训练。")
    # ===============================================================================

    # 修改迭代范围，确保从 start_epoch 开始
    for epoch in range(start_epoch, ns.epochs):
        feed(net, optimizer, x, True, True, batch, epoch)
        feed(net, optimizer, x_test, False, False, batch, epoch)
        
        # 保存周期性检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
        }, ns.checkpoint_path)
        print(f"Epoch {epoch} 完成，训练状态已保存至: {ns.checkpoint_path}")