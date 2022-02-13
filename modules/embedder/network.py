import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from time import time

import cv2
import math
import config_path as path
from utils import overlap_ratio
import matplotlib.pyplot as plt
from roi_align import RoIAlign


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                         bias=False)


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
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
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # block4

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
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
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


def resnet50(model_path=None):
    model = ResNet(Bottleneck, [3, 4, 6, 3])

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


class Embedder(nn.Module):
    def __init__(self, mse=False):
        super(Embedder, self).__init__()
        self.mse = mse

        self.feature = resnet18(path.res_model)  # 512 -> 11x11

        self.fc3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.3),
            nn.Conv2d(256, 128, 1),
            nn.GroupNorm(4, 128),
            nn.ReLU(inplace=True),
        )

        self.fc4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.3),
            nn.Conv2d(512, 256, 1),
            nn.GroupNorm(4, 256),
            nn.ReLU(inplace=True),
        )

        self.roi = RoIAlign(128, 128).cuda()

        self.init2_flag = False
        self.update2_flag = False
        self.update3_flag = True

    def init1(self, img, box):  # [x y w h]
        bbox = box.copy()
        self.load_params()

        with torch.no_grad():
            bbox[2:] = bbox[2:] + bbox[:2]
            bbox = bbox.astype(int)
            tmp = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            tmp = cv2.resize(tmp, (128, 128))

            img = self.process(tmp)
            img = torch.Tensor(img).cuda()

            a3, a4 = self.feature(img)  # 512 -> 11x11
            a = torch.cat([self.fc3(a3).view(1, -1),
                           self.fc4(a4).view(1, -1)], dim=1)
            self.a1 = F.normalize(a, p=2, dim=1)

            tmp2 = tmp[:, ::-1, :].copy()
            img = self.process(tmp2)
            img = torch.Tensor(img).cuda()

            a3, a4 = self.feature(img)  # 512 -> 11x11
            a = torch.cat([self.fc3(a3).view(1, -1),
                           self.fc4(a4).view(1, -1)], dim=1)
            self.a2 = F.normalize(a, p=2, dim=1)  # LT
            self.a3 = F.normalize(a, p=2, dim=1)  # ST
            self.a4 = F.normalize(a, p=2, dim=1)


    def init2(self, img, box):  # [x y x y]
        bbox = box.copy()

        bbox = np.clip(bbox, 0, 9999)
        with torch.no_grad():
            bbox = bbox.astype(int)
            tmp = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            tmp = cv2.resize(tmp, (128, 128))
            # tmp = tmp[:, ::-1, :]

            img = self.process(tmp)
            img = torch.Tensor(img).cuda()

            a3, a4 = self.feature(img)  # 512 -> 11x11
            a = torch.cat([self.fc3(a3).view(1, -1),
                           self.fc4(a4).view(1, -1)], dim=1)
            self.a2 = F.normalize(a, p=2, dim=1)

    def inference1(self, img, boxes, scores, thres=0.5):  # [x y x y]
        """
        without Update
        """
        boxes = boxes.reshape(-1, 4)
        scores = np.array([scores]).reshape(-1)
        with torch.no_grad():

            img = self.process(img)
            img = torch.Tensor(img).cuda()
            _boxes = torch.Tensor(boxes).cuda()
            _box_index = torch.zeros(_boxes.size(0), dtype=torch.int).cuda()

            patch = self.roi(img, _boxes, _box_index)

            b = patch.size(0)
            p3, p4 = self.feature(patch)
            p = torch.cat([self.fc3(p3).view(b, -1),
                           self.fc4(p4).view(b, -1)], dim=1)
            p = F.normalize(p, p=2, dim=1)

            a = self.a1.expand(b, self.a1.size(1))

            dist = F.mse_loss(p, a, reduction='none').sum(1)
            dist = dist.data.cpu().numpy()

            idx = np.argmin(dist)
            _box = boxes[idx]
            _dist = dist[idx]
            _scores = scores[idx]

            if _dist > thres:
                find = False
            else:
                find = True

            return _box, _scores, find

    def inference2(self, img, boxes, scores, frame_idx, thres=0.5):  # [x y x y]
        boxes = boxes.reshape(-1, 4)
        scores = np.array([scores]).reshape(-1)
        with torch.no_grad():

            img = self.process(img)
            img = torch.Tensor(img).cuda()
            _boxes = torch.Tensor(boxes).cuda()
            _box_index = torch.zeros(_boxes.size(0), dtype=torch.int).cuda()

            patch = self.roi(img, _boxes, _box_index)

            b = patch.size(0)
            p3, p4 = self.feature(patch)
            p = torch.cat([self.fc3(p3).view(b, -1),
                           self.fc4(p4).view(b, -1)], dim=1)
            p = F.normalize(p, p=2, dim=1)

            a1 = self.a1.expand(b, self.a1.size(1))
            a2 = self.a2.expand(b, self.a2.size(1))
            a3 = self.a3.expand(b, self.a3.size(1))

            dist1 = F.mse_loss(p, a1, reduction='none').sum(1)
            dist1 = dist1.data.cpu().numpy()

            dist2 = F.mse_loss(p, a2, reduction='none').sum(1)
            dist2 = dist2.data.cpu().numpy()

            dist3 = F.mse_loss(p, a3, reduction='none').sum(1)
            dist3 = dist3.data.cpu().numpy()

            dist = dist1 * 0.5 + dist2 * 0.3 + dist3 * 0.2

            idx = np.argmin(dist)
            _box = boxes[idx]
            _dist = dist[idx]
            _scores = scores[idx]

            if frame_idx % 300 == 0:
                self.update3_flag = True

            if frame_idx % 1000 == 0:
                self.update2_flag = True

            if _dist > thres:
                find = False
            else:
                find = True

                if self.update3_flag:
                    self.update3_flag = False

                    if (
                            F.mse_loss(self.a1, p[idx].unsqueeze(0), reduction='none').sum() * 0.5
                            + F.mse_loss(self.a2, p[idx].unsqueeze(0), reduction='none').sum() * 0.5
                            <
                            F.mse_loss(self.a1, self.a3, reduction='none').sum() * 0.5
                            + F.mse_loss(self.a2, self.a3, reduction='none').sum() * 0.5
                    ):

                        if self.update2_flag:
                            self.update2_flag = False
                            self.a2 = self.a3.clone().detach()

                        self.a3 = p[idx].unsqueeze(0).clone().detach()

            return _box, _scores, find

    def inference2ox(self, img, boxes, scores, frame_idx, thres=0.5):  # [x y x y]
        boxes = boxes.reshape(-1, 4)
        scores = np.array([scores]).reshape(-1)
        with torch.no_grad():

            img = self.process(img)
            img = torch.Tensor(img).cuda()
            _boxes = torch.Tensor(boxes).cuda()
            _box_index = torch.zeros(_boxes.size(0), dtype=torch.int).cuda()

            patch = self.roi(img, _boxes, _box_index)

            b = patch.size(0)
            p3, p4 = self.feature(patch)
            p = torch.cat([self.fc3(p3).view(b, -1),
                           self.fc4(p4).view(b, -1)], dim=1)
            p = F.normalize(p, p=2, dim=1)

            a1 = self.a1.expand(b, self.a1.size(1))
            a2 = self.a2.expand(b, self.a2.size(1))
            a3 = self.a3.expand(b, self.a3.size(1))

            dist1 = F.mse_loss(p, a1, reduction='none').sum(1)
            dist1 = dist1.data.cpu().numpy()

            dist2 = F.mse_loss(p, a2, reduction='none').sum(1)
            dist2 = dist2.data.cpu().numpy()

            dist3 = F.mse_loss(p, a3, reduction='none').sum(1)
            dist3 = dist3.data.cpu().numpy()

            dist = dist1 * 0.5 + dist2 * 0.3 + dist3 * 0.2

            idx = np.argmin(dist)
            _box = boxes[idx]
            _dist = dist[idx]
            _scores = scores[idx]

            if frame_idx % 300 == 0:
                self.update3_flag = True

            if frame_idx % 1000 == 0:
                self.update2_flag = True

            if _dist > thres:
                find = False
            else:
                find = True

                if self.update3_flag:
                    self.update3_flag = False

                    if (
                            F.mse_loss(self.a1, p[idx].unsqueeze(0), reduction='none').sum() * 0.5
                            + F.mse_loss(self.a2, p[idx].unsqueeze(0), reduction='none').sum() * 0.5
                            <
                            F.mse_loss(self.a1, self.a3, reduction='none').sum() * 0.5
                            + F.mse_loss(self.a2, self.a3, reduction='none').sum() * 0.5
                    ):

                        if self.update2_flag:
                            self.update2_flag = False
                            self.a2 = self.a3.clone().detach()

                        self.a3 = p[idx].unsqueeze(0).clone().detach()

            return _box, _scores, _dist, find

    def inference3(self, img, boxes, scores, frame_idx, thres=0.5):  # ST
        boxes = boxes.reshape(-1, 4)
        scores = np.array([scores]).reshape(-1)
        with torch.no_grad():

            img = self.process(img)
            img = torch.Tensor(img).cuda()
            _boxes = torch.Tensor(boxes).cuda()
            _box_index = torch.zeros(_boxes.size(0), dtype=torch.int).cuda()

            patch = self.roi(img, _boxes, _box_index)

            b = patch.size(0)
            p3, p4 = self.feature(patch)
            p = torch.cat([self.fc3(p3).view(b, -1),
                           self.fc4(p4).view(b, -1)], dim=1)
            p = F.normalize(p, p=2, dim=1)

            a1 = self.a1.expand(b, self.a1.size(1))
            a3 = self.a3.expand(b, self.a3.size(1))

            dist1 = F.mse_loss(p, a1, reduction='none').sum(1)
            dist1 = dist1.data.cpu().numpy()

            dist3 = F.mse_loss(p, a3, reduction='none').sum(1)
            dist3 = dist3.data.cpu().numpy()

            dist = dist1 * 0.8 + dist3 * 0.2

            idx = np.argmin(dist)
            _box = boxes[idx]
            _dist = dist[idx]
            _scores = scores[idx]

            if frame_idx % 300 == 0:
                self.update3_flag = True

            if frame_idx % 1000 == 0:
                self.update2_flag = True

            if _dist > thres:
                find = False
            else:
                find = True

                if self.update3_flag:
                    self.update3_flag = False

                    if (
                            F.mse_loss(self.a1, p[idx].unsqueeze(0), reduction='none').sum() * 0.5
                            + F.mse_loss(self.a2, p[idx].unsqueeze(0), reduction='none').sum() * 0.5
                            <
                            F.mse_loss(self.a1, self.a3, reduction='none').sum() * 0.5
                            + F.mse_loss(self.a2, self.a3, reduction='none').sum() * 0.5
                    ):

                        if self.update2_flag:
                            self.update2_flag = False
                            self.a2 = self.a3.clone().detach()

                        self.a3 = p[idx].unsqueeze(0).clone().detach()

            return _box, _scores, find

    def inference4(self, img, boxes, scores, frame_idx, thres=0.5):  # LT
        boxes = boxes.reshape(-1, 4)
        scores = np.array([scores]).reshape(-1)
        with torch.no_grad():

            img = self.process(img)
            img = torch.Tensor(img).cuda()
            _boxes = torch.Tensor(boxes).cuda()
            _box_index = torch.zeros(_boxes.size(0), dtype=torch.int).cuda()

            patch = self.roi(img, _boxes, _box_index)

            b = patch.size(0)
            p3, p4 = self.feature(patch)
            p = torch.cat([self.fc3(p3).view(b, -1),
                           self.fc4(p4).view(b, -1)], dim=1)
            p = F.normalize(p, p=2, dim=1)

            a1 = self.a1.expand(b, self.a1.size(1))
            a2 = self.a2.expand(b, self.a2.size(1))

            dist1 = F.mse_loss(p, a1, reduction='none').sum(1)
            dist1 = dist1.data.cpu().numpy()

            dist2 = F.mse_loss(p, a2, reduction='none').sum(1)
            dist2 = dist2.data.cpu().numpy()

            dist = dist1 * 0.7 + dist2 * 0.3

            idx = np.argmin(dist)
            _box = boxes[idx]
            _dist = dist[idx]
            _scores = scores[idx]

            if frame_idx % 300 == 0:
                self.update3_flag = True

            if frame_idx % 1000 == 0:
                self.update2_flag = True

            if _dist > thres:
                find = False
            else:
                find = True

                if self.update3_flag:
                    self.update3_flag = False

                    if (
                            F.mse_loss(self.a1, p[idx].unsqueeze(0), reduction='none').sum() * 0.5
                            + F.mse_loss(self.a2, p[idx].unsqueeze(0), reduction='none').sum() * 0.5
                            <
                            F.mse_loss(self.a1, self.a3, reduction='none').sum() * 0.5
                            + F.mse_loss(self.a2, self.a3, reduction='none').sum() * 0.5
                    ):

                        if self.update2_flag:
                            self.update2_flag = False
                            self.a2 = self.a3.clone().detach()

                        self.a3 = p[idx].unsqueeze(0).clone().detach()

            return _box, _scores, find


    def process(self, img):  # BGR
        img = img[:, :, ::-1]  # RGB
        img = img.astype(np.float32) / 255
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

        return img.transpose(2, 0, 1)[None, :, :, :]

    def loss_fun(self, a, p, n):
        b = a.size(0)

        a3, a4 = self.feature(a)
        a = torch.cat([self.fc3(a3).view(b, -1),
                       self.fc4(a4).view(b, -1)], dim=1)
        a = F.normalize(a, p=2, dim=1)

        p3, p4 = self.feature(p)
        p = torch.cat([self.fc3(p3).view(b, -1),
                       self.fc4(p4).view(b, -1)], dim=1)
        p = F.normalize(p, p=2, dim=1)

        n3, n4 = self.feature(n)
        n = torch.cat([self.fc3(n3).view(b, -1),
                       self.fc4(n4).view(b, -1)], dim=1)
        n = F.normalize(n, p=2, dim=1)

        pos_dist = F.mse_loss(p, a, reduction='none').sum(1)
        neg_dist = F.mse_loss(n, a, reduction='none').sum(1)
        loss = F.relu(pos_dist - neg_dist + 1) / b

        return loss.sum()

    def load_params(self, model_path=path.embed_model):
        # print('==> from \'{}\' load params'.format(model_path))
        pretrain_dict = torch.load(model_path)
        pretrain_dict = pretrain_dict['params']
        tmp_dict = self.state_dict()
        for key, value in pretrain_dict.items():
            if key in tmp_dict:
                # print('load pretrain params: %s'%key)
                tmp_dict[key] = value
        self.load_state_dict(tmp_dict)

    # ================================================================================================================ #


if __name__ == '__main__':
    print('feature embedding')
