import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from data.dataset import LaSOT2
from modules.guider.network import Guider
from modules.guider.lmdb_patch import build_dataset

project_fir = '/data2/Documents/Workspce/PAMI2020_train/modules/guider'

if not os.path.exists(os.path.join(project_fir, 'log_mse')):
    os.mkdir(os.path.join(project_fir, 'log_mse'))
writer1 = SummaryWriter(os.path.join(project_fir, 'log_mse', 'plot1'))
writer2 = SummaryWriter(os.path.join(project_fir, 'log_mse', 'plot2'))
writer3 = SummaryWriter(os.path.join(project_fir, 'log_mse', 'image'))


if __name__ == '__main__':

    print('=> Creating model...')
    start_epoch = 0
    guided = Guider(mse=True)
    guided = guided.cuda()
    optimizer = torch.optim.Adam(
        [
            {'params': guided.feature.layer4.parameters()},
            {'params': guided.deconv.parameters()},
            {'params': guided.fc_z.parameters()},
            {'params': guided.fc_zk.parameters()},
        ],
        lr=1e-4, weight_decay=1e-4)  # 1e-6

    start_niter = 0
    best_loss = 999

    # checkpoint = torch.load(os.path.join(project_fir, 'last_mse.pth'))
    # # optimizer.load_state_dict(checkpoint['optimizer'])
    # guided.load_state_dict(checkpoint['params'])
    # start_epoch = checkpoint['epoch']
    # start_niter = checkpoint['niter']
    # best_loss = checkpoint['best_loss']
    # print('=> Loading from... epoch {}'.format(start_epoch))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 5e-5

    print('=> Loading data...')
    trainset, valset = build_dataset()
    train_loader = DataLoader(
        trainset,  #LaSOT2('train'),
        batch_size=64,
        num_workers=8,
        shuffle=True,
        drop_last=True)

    val_loader = DataLoader(
        valset,  # LaSOT2('test'),
        batch_size=16,
        num_workers=8,
        shuffle=True,
        drop_last=True)

    print('=> Training... from {} epoch'.format(start_epoch+1))
    best_epoch = 1
    back = 0
    sub_batch = 0
    niter = start_niter
    loss_sum = []
    for epoch in range(start_epoch, 1000):

        guided.train()
        for j, batch in enumerate(train_loader):

            # z = batch['z'].cuda()
            # s = batch['s'].cuda()
            # t = batch['t'].cuda()
            z = batch[0].cuda()
            s = batch[1].cuda()
            t = batch[2].cuda()

            out = guided.search(s, z)
            loss = guided.loss_fun(out, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            niter += 1

            if epoch >= 0:
                loss_sum.append(loss.item())

                if niter % 5 == 0:
                    print(epoch, 'Iter {:0>2d}/{} | Loss: {:.9f}'.format(niter, len(train_loader), np.mean(loss_sum)))
                    writer1.add_scalar('Loss', np.mean(loss_sum), niter)
                    loss_sum = []
            else:
                print(epoch, 'Iter {:0>2d}/{} | Loss: {:.9f}'.format(niter, len(train_loader), loss.item()))

        out_grid = make_grid(torch.sigmoid(out[:16]), nrow=8, pad_value=1).unsqueeze(0)
        t_grid = make_grid(t[:16], nrow=8, pad_value=1).unsqueeze(0)
        grid = torch.cat([out_grid, t_grid], dim=0)
        grid = make_grid(grid, nrow=1, pad_value=1)
        writer3.add_image('train/image', grid, global_step=niter)

        val_loss_sum = []
        guided.eval()
        for j, batch in enumerate(tqdm(val_loader)):

            # z = batch['z'].cuda()
            # s = batch['s'].cuda()
            # t = batch['t'].cuda()
            z = batch[0].cuda()
            s = batch[1].cuda()
            t = batch[2].cuda()

            with torch.no_grad():
                out = guided.search(s, z)
                loss = guided.loss_fun(out, t)

                val_loss_sum.append(loss.item())

        out_grid = make_grid(torch.sigmoid(out), nrow=8, pad_value=1).unsqueeze(0)
        t_grid = make_grid(t, nrow=8, pad_value=1).unsqueeze(0)
        grid = torch.cat([out_grid, t_grid], dim=0)
        grid = make_grid(grid, nrow=1, pad_value=1)
        writer3.add_image('val/image', grid, global_step=niter)

        save_dict = dict()
        save_dict['params'] = guided.state_dict()
        save_dict['optimizer'] = optimizer.state_dict()
        save_dict['epoch'] = epoch + 1
        save_dict['niter'] = niter + 1
        save_dict['best_loss'] = best_loss
        torch.save(save_dict, os.path.join(project_fir, 'last_mse_{:0>3d}.pth'.format(epoch)))
        if np.mean(val_loss_sum) < best_loss:
            best_loss = np.mean(val_loss_sum)
            best_epoch = epoch + 1
            torch.save(save_dict, os.path.join(project_fir, 'best_mse.pth'))
        print('Epoch {:0>2d} | Best: {} | loss: {}'.format(epoch+1, best_epoch, np.mean(val_loss_sum)))
        if epoch > 4:
            writer2.add_scalar('Loss', np.mean(val_loss_sum), niter)
