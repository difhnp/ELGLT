import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

import cv2
import math
import config_path as path
from utils import overlap_ratio


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation) # addd dilation
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0])  # block1
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # block2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # block3
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)  # block4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)

        return x3, x4


def resnet18(model_path=None):
    model = ResNet(BasicBlock, [2, 2, 2, 2])

    if model_path is not None:
        pretrain_dict = torch.load(model_path)
        tmp_dict = model.state_dict()
        for key, value in pretrain_dict.items():
            if key in tmp_dict:
                # print('load pretrain params: %s'%key)
                tmp_dict[key] = value
        model.load_state_dict(tmp_dict)

    return model

# ==================================================================================================================== #


class Guider(nn.Module):
    def __init__(self, mse=False):
        super(Guider, self).__init__()
        self.mse = mse

        self.feature = resnet18(path.res_model)  # 512 -> 11x11

        self.fc_z = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.3),
            nn.Conv2d(512, 512, 1),
            nn.Sigmoid(),
        )

        self.fc_zk = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.3),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.Dropout2d(0.5),

            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, output_padding=0, bias=False),  # 11 -> 22
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Dropout2d(0.3),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=0, bias=False),  # 22 -> 44
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64,  16, 4, stride=2, padding=1, output_padding=0, bias=False),  # 44 -> 88
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def search(self, s, z):
        b = s.size(0)

        _, s = self.feature(s)  # 512 -> 11x11

        _, _z = self.feature(z)  # 512 -> 11x11

        z = self.fc_z(_z)
        zk = self.fc_zk(_z)

        z = z.expand(b, z.size(1), s.size(2), s.size(3))
        zk = zk.expand(b, zk.size(1), s.size(2), s.size(3))

        s = s * z
        s = torch.cat([s, zk], dim=1)
        s = self.deconv(s)

        return s

    def init(self, img, box):  # [x y w h]
        bbox = box.copy()
        self.load_params()

        with torch.no_grad():
            center_pos = bbox[:2] + bbox[2:] / 2

            rand_scale = 0.5  # np.random.uniform(0.45, 0.55)
            w_z = bbox[2] + rand_scale * np.sum(bbox[2:])
            h_z = bbox[3] + rand_scale * np.sum(bbox[2:])
            s_z = np.sqrt(w_z * h_z)

            patch = self.get_subwindow(img, center_pos, 127, s_z, np.mean(img, axis=(0, 1)))

            patch = self.process(patch)
            patch = torch.Tensor(patch).cuda()

            _, z = self.feature(patch)
            self.z = self.fc_z(z)
            self.zk = self.fc_zk(z)

    def inference(self, img):
        h, w, _ = img.shape
        with torch.no_grad():
            img = cv2.resize(img, (512, 512))
            tmp = img.copy()
            img = self.process(img)
            img = torch.Tensor(img).cuda()

            _, s = self.feature(img)  # 512 -> 11x11

            z = self.z.expand(1, self.z.size(1), s.size(2), s.size(3))
            zk = self.zk.expand(1, self.zk.size(1), s.size(2), s.size(3))

            s = s * z
            s = torch.cat([s, zk], dim=1)
            s = self.deconv(s)

            s = torch.sigmoid(s)

            s = s.data[0, 0].cpu().numpy()
            s = cv2.resize(s, (w, h))
        return s

    def process(self, img):  # BGR
        img = img[:, :, ::-1]
        img = img.astype(np.float32) / 255
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

        return img.transpose(2, 0, 1)[None, :, :, :]

    def loss_fun(self, s, t):
        if self.mse:
            s = torch.sigmoid(s)
            w = t.detach().clone()
            w[w < 0.05] = 0.05
            loss = (F.mse_loss(s, t, reduction='none') * w).sum() / s.numel()
        else:
            loss = F.binary_cross_entropy_with_logits(s, t, reduction='mean', pos_weight=torch.tensor(20))
        return loss

    def load_params(self, model_path=path.guid_model):
        # print('==> from \'{}\' load params'.format(model_path))
        pretrain_dict = torch.load(model_path)
        pretrain_dict = pretrain_dict['params']
        tmp_dict = self.state_dict()
        for key, value in pretrain_dict.items():
            if key in tmp_dict:
                # print('load pretrain params: %s'%key)
                tmp_dict[key] = value
        self.load_state_dict(tmp_dict)

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))

        return im_patch

# ==================================================================================================================== #


if __name__ == '__main__':
    guided = Guider().cuda()
    guided.eval()

    z = torch.rand([16, 3, 127, 127], dtype=torch.float32).cuda()
    x = torch.rand([16, 3, 512, 512], dtype=torch.float32).cuda()

    with torch.no_grad():
        x = guided.search(x, z)
        print('')
