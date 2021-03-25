from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='BCCD', 
                    type=str, help='VOC or BCCD')
parser.add_argument('--batch_size', default=12, type=int,
                    help='Batch size for training')
parser.add_argument('--model', default='ssd300_mAP_77.43_v2.pth', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--max_iter', default=200, type=int,
                    help='max_iter')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

# GPUの設定
import torch
torch.cuda.is_available() 
torch.set_default_tensor_type('torch.cuda.FloatTensor')  
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# 訓練データの読み込み
cfg = voc
dataset = VOCDetection(root=VOC_ROOT,
                       transform=SSDAugmentation(cfg['min_dim'],
                                                 MEANS))
# ネットワークの定義
ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
net = ssd_net.to(device)
net = torch.nn.DataParallel(ssd_net)
cudnn.benchmark = True

# パラメータロード
ssd_net.load_weights(str(args.save_folder) + str(args.model))

# 最適化パラメータの設定
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)

# 損失関数の設定
criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                         False, args.cuda)

# データセットの読み込み
data_loader = data.DataLoader(dataset, args.batch_size,
                              shuffle=True, collate_fn=detection_collate,
                              pin_memory=True)

# 学習の開始
net.train()
batch_iterator = None
epoch_size = len(dataset) // args.batch_size

for iteration in range(args.max_iter):   
    if (not batch_iterator) or (iteration % epoch_size ==0):
        batch_iterator = iter(data_loader)
        loc_loss = 0
        conf_loss = 0
        
    # 訓練データをバッチで読み込みGPUへ転送
    images, targets = next(batch_iterator)    
    images = images.to(device)
    targets = [ann.to(device) for ann in targets]

    # 順伝播
    t0 = time.time()
    out = net(images)

    # 逆伝播
    optimizer.zero_grad()
    loss_l, loss_c = criterion(out, targets)
    loss = loss_l + loss_c
    loss.backward()
    optimizer.step()
    t1 = time.time()
    loc_loss += loss_l.item()
    conf_loss += loss_c.item()
    
    #ログの出力
    if iteration % 10 == 0:
        print('timer: %.4f sec.' % (t1 - t0))
        print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

# 学習済みモデルの保存
torch.save(ssd_net.state_dict(),
           str(args.save_folder) + '' + str(args.dataset) + '.pth')