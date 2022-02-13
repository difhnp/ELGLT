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

from data.dataset import LaSOT3
from modules.embedder.network import Embedder
from modules.embedder.lmdb_patch import build_dataset

project_fir = '/data2/Documents/Workspce/PAMI2020_train/modules/embedder'

if not os.path.exists(os.path.join(project_fir, 'log_em')):
    os.mkdir(os.path.join(project_fir, 'log_em'))
writer1 = SummaryWriter(os.path.join(project_fir, 'log_em', 'plot1'))
writer2 = SummaryWriter(os.path.join(project_fir, 'log_em', 'plot2'))
writer3 = SummaryWriter(os.path.join(project_fir, 'log_em', 'image'))


if __name__ == '__main__':

    print('=> Creating model...')
    start_epoch = 0
    verifier = Embedder()
    verifier = verifier.cuda()
    optimizer = torch.optim.Adam(
        [
            {'params': verifier.feature.layer3.parameters(), 'lr': 5e-5},
            {'params': verifier.feature.layer4.parameters(), 'lr': 5e-5},
            {'params': verifier.fc3.parameters()},
            {'params': verifier.fc4.parameters()},
        ],
        lr=1e-4, weight_decay=1e-4)  # 1e-6

    start_niter = 0
    best_loss = 999

    # checkpoint = torch.load(os.path.join(project_fir, 'last_em.pth'))
    # # optimizer.load_state_dict(checkpoint['optimizer'])
    # verifier.load_state_dict(checkpoint['params'])
    # start_epoch = checkpoint['epoch']
    # start_niter = checkpoint['niter']
    # best_loss = checkpoint['best_loss']
    # print('=> Loading from... epoch {}'.format(start_epoch))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 5e-5

    trainset, valset = build_dataset()
    print('=> Loading data...')
    train_loader = DataLoader(
        trainset,  #LaSOT3('train'),
        batch_size=64,
        num_workers=6,
        shuffle=True,
        drop_last=True)

    val_loader = DataLoader(
        valset,  # LaSOT3('test'),
        batch_size=64,
        num_workers=6,
        shuffle=False,
        drop_last=False)

    print('=> Training... from {} epoch'.format(start_epoch+1))
    best_epoch = 1
    back = 0
    sub_batch = 0
    niter = start_niter
    loss_sum = []
    for epoch in range(start_epoch, 1000):

        verifier.train()
        for j, batch in enumerate(train_loader):

            # a = batch['a'].cuda()
            # p = batch['p'].cuda()
            # n = batch['n'].cuda()

            a = batch[0].cuda()
            p = batch[1].cuda()
            n = batch[2].cuda()

            loss = verifier.loss_fun(a, p, n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            niter += 1

            if epoch + 1 > 0:
                loss_sum.append(loss.item())

                if niter % 1 == 0:
                    print('Iter {:0>2d} | Loss: {:.9f}'.format(niter, np.mean(loss_sum)))
                    writer1.add_scalar('Loss', np.mean(loss_sum), niter)
                    loss_sum = []

        val_loss_sum = []
        verifier.eval()
        for j, batch in enumerate(tqdm(val_loader)):
            # a = batch['a'].cuda()
            # p = batch['p'].cuda()
            # n = batch['n'].cuda()

            a = batch[0].cuda()
            p = batch[1].cuda()
            n = batch[2].cuda()

            with torch.no_grad():
                loss = verifier.loss_fun(a, p, n)

                val_loss_sum.append(loss.item())

        save_dict = dict()
        save_dict['params'] = verifier.state_dict()
        save_dict['optimizer'] = optimizer.state_dict()
        save_dict['epoch'] = epoch + 1
        save_dict['niter'] = niter + 1
        save_dict['best_loss'] = best_loss
        torch.save(save_dict, os.path.join(project_fir, 'last_em2_{:0>3d}.pth'.format(epoch)))
        if np.mean(val_loss_sum) < best_loss:
            best_loss = np.mean(val_loss_sum)
            best_epoch = epoch + 1
            torch.save(save_dict, os.path.join(project_fir, 'best_em2.pth'))
        print('Epoch {:0>2d} | Best: {} | loss: {}'.format(epoch+1, best_epoch, np.mean(val_loss_sum)))
        if epoch + 1 > 0:
            writer2.add_scalar('Loss', np.mean(val_loss_sum), niter)
