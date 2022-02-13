import cv2
import random
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import lmdb

import torch
from modules.embedder.dataset import SubSet, BaseDataset
import modules.embedder.set_paths as paths

LMDB_PATH = {
    'coco_train': paths.lmdb_coco_train,  # 860001
    'coco_val': paths.lmdb_coco_val,
    'coco_mask_train': paths.lmdb_coco_mask_train,
    'coco_mask_val': paths.lmdb_coco_mask_val,
    'got10k_train': paths.lmdb_got10k_train,
    'got10k_train_vot': paths.lmdb_got10k_train,  # 8335
    'got10k_val': paths.lmdb_got10k_val,
    'lasot_train': paths.lmdb_lasot_train,  # 1120
    'lasot_val': paths.lmdb_lasot_val,
    'trackingnet_train_p0': paths.lmdb_trackingnet_train_p0,  # 10044
    'trackingnet_train_p1': paths.lmdb_trackingnet_train_p1,  #
    'trackingnet_train_p2': paths.lmdb_trackingnet_train_p2,  #
}

JSON_PATH = {
    'coco_train': paths.json_lmdb_coco_train,
    'coco_val': paths.json_lmdb_coco_val,
    'coco_mask_train': paths.json_lmdb_coco_train,
    'coco_mask_val': paths.json_lmdb_coco_val,
    'got10k_train': paths.json_lmdb_got10k_train,
    'got10k_train_vot': paths.json_lmdb_got10k_train_vot,
    'got10k_val': paths.json_lmdb_got10k_val,
    'lasot_train': paths.json_lmdb_lasot_train,
    'lasot_val': paths.json_lmdb_lasot_val,
    'trackingnet_train_p0': paths.json_lmdb_trackingnet_train_p0,
    'trackingnet_train_p1': paths.json_lmdb_trackingnet_train_p1,
    'trackingnet_train_p2': paths.json_lmdb_trackingnet_train_p2,
}


def my_collate(batch):
    template_img = [torch.Tensor(item[0]).unsqueeze(0) for item in batch]
    search_img = [torch.Tensor(item[1]).unsqueeze(0) for item in batch]
    s_box = [torch.Tensor(item[2]).unsqueeze(0) for item in batch]

    return [template_img, search_img, s_box]


class VideoDataset(BaseDataset):

    def __init__(self, name_list: list = None, num_sample: int = None, aug_dict: dict = None):
        super().__init__()

        self.debug = False
        if self.debug:
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3)

        self.data_path: dict = LMDB_PATH
        self.json_path: dict = JSON_PATH

        self.sample_range: int = 200

        self.aug_color = None
        self.aug_gray = None
        self.aug_flip = None
        self.update_hyper_params(aug_dict)

        # load dataset
        self.LMDB_ENVS = {}
        self.LMDB_HANDLES = {}
        self.video_list: List = []
        for name in name_list:
            env = lmdb.open(self.data_path[name], readonly=True, lock=False, readahead=False, meminit=False)
            self.LMDB_ENVS[name] = env
            item = env.begin(write=False)
            self.LMDB_HANDLES[name] = item

            dataset = SubSet(name=name, load=self.json_path[name])
            if 'coco' in name:
                random.shuffle(dataset.data_set)
                self.video_list += dataset.data_set[:10000]
            else:
                self.video_list += dataset.data_set

        # repeat and shuffle
        if len(name_list) > 1:
            random.shuffle(self.video_list)
        while len(self.video_list) < num_sample:
            self.video_list += self.video_list
        self.video_list = self.video_list[:num_sample]
        self.video_num = len(self.video_list)
        random.shuffle(self.video_list)

    def __len__(self):
        return self.video_num

    def __getitem__(self, item):
        a_dict, p_dict = self.check_sample(self.video_list[item], self.video_list, self.sample_range)
        # read RGB image, [x y w h]
        a_img, a_box = self.parse_frame_lmdb(a_dict, self.LMDB_HANDLES)
        p_img, p_box = self.parse_frame_lmdb(p_dict, self.LMDB_HANDLES)

        a_box, p_box = map(lambda x: x.astype(int), [a_box, p_box])

        a_img, a_box = self.crop_patch(
            a_img, a_box,
            out_size=127,
            scale_factor=2,
            jitter_f=[0, 0])

        p_img, p_box = self.crop_patch(
            p_img, p_box,
            out_size=512,
            scale_factor=3 if np.random.rand() > 0.5 else 5,
            jitter_f=[0.2, 2])

        target = self.gauss_map(p_box[0] + p_box[2] / 2., p_box[1] + p_box[3] / 2., p_box[2], p_box[3], 88, 88)

        if np.random.rand() < self.aug_color:
            a_img, p_img = map(self.color_jitter, [a_img, p_img])
        if np.random.rand() < self.aug_gray:
            a_img, p_img = map(self.color_gray, [a_img, p_img])

        if np.random.rand() < self.aug_flip:
            a_img = self.horiz_flip(a_img)
        if np.random.rand() < self.aug_flip:
            p_img = self.horiz_flip(p_img)
            target = self.horiz_flip(target)

        if self.debug:
            print(a_box, p_box)
            self.debug_fn([a_img, p_img, target], [a_box, p_box])

        a_img, p_img = map(lambda x: x.transpose(2, 0, 1).astype(np.float32), [a_img, p_img])

        return a_img, p_img, target.astype(np.float32)[None, :, :]

    def gauss_map(self, cx, cy, w, h, img_w, img_h):
        if w == 0 or h == 0:
            return np.zeros([img_h, img_w])

        mu_w = cx * 88 / 512
        mu_h = cy * 88 / 512

        # [-n sigma, +n sigma]
        sigma_w = w * 0.2 * 88 / 512
        sigma_h = h * 0.2 * 88 / 512
        #    sigma_w = 2
        #    sigma_h = 2

        x_w = np.linspace(0, img_w - 1, img_w)
        x_h = np.linspace(0, img_h - 1, img_h)

        y_w = np.exp(-((x_w - mu_w) ** 2) / (2 * sigma_w ** 2)) / (sigma_w * np.sqrt(2 * np.pi))
        y_h = np.exp(-((x_h - mu_h) ** 2) / (2 * sigma_h ** 2)) / (sigma_h * np.sqrt(2 * np.pi))

        y_w /= np.max(y_w)
        y_h /= np.max(y_h)

        y_w = np.matrix(y_w.reshape(1, len(y_w)))
        y_h = np.matrix(y_h.reshape(len(y_h), 1))

        #    plt.plot(np.array(y_w).reshape(-1))

        gauss_map = y_h * y_w
        gauss_map = np.array(gauss_map)
        gauss_map /= np.max(gauss_map)

        return gauss_map

    def debug_fn(self, im, box):  # [x, y, x, y]
        a_img = im[0]
        p_img = im[1]
        t_img = im[2]

        a_box = box[0]
        p_box = box[1]

        a_img = cv2.rectangle(
            a_img,
            (int(a_box[0]), int(a_box[1])),
            (int(a_box[0]+a_box[2]-1), int(a_box[1]+a_box[3]-1)), (0, 255, 0), 4)

        p_img = cv2.rectangle(
            p_img,
            (int(p_box[0]), int(p_box[1])),
            (int(p_box[0]+p_box[2]-1), int(p_box[1]+p_box[3]-1)), (0, 255, 0), 4)

        self.ax1.imshow(a_img)
        self.ax2.imshow(p_img)
        self.ax3.imshow(t_img)
        self.fig.show()
        plt.waitforbuttonpress()


def build_dataset():
    train_dataset = VideoDataset(name_list=['got10k_train', 'lasot_train'], num_sample=30000, aug_dict={
        'aug_color': 1,
        'aug_gray': 0.05,
        'aug_flip': 0.5,
    })
    val_dataset = VideoDataset(name_list=['got10k_val'], num_sample=3000, aug_dict={
        'aug_color': 0,
        'aug_gray': 0,
        'aug_flip': 0,
    })

    return train_dataset, val_dataset


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    trainset, valset = build_dataset()

    train_loader = DataLoader(
        valset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )

    for i, image in enumerate(train_loader):
        print(i)
