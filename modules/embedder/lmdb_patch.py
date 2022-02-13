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
        _, n_dict = self.check_sample(self.video_list[(item + np.random.randint(10, 100)) % self.video_num],
                                      self.video_list, self.sample_range)

        # read RGB image, [x y w h]
        a_img, a_box = self.parse_frame_lmdb(a_dict, self.LMDB_HANDLES)
        p_img, p_box = self.parse_frame_lmdb(p_dict, self.LMDB_HANDLES)
        n_img, n_box = self.parse_frame_lmdb(n_dict, self.LMDB_HANDLES)

        a_box, p_box, n_box = map(lambda x: x.astype(int), [a_box, p_box, n_box])

        a_box[2:] = a_box[2:] + a_box[:2]  # [x y x y]
        p_box[2:] = p_box[2:] + p_box[:2]  # [x y x y]
        n_box[2:] = n_box[2:] + n_box[:2]  # [x y x y]

        a_img = a_img[a_box[1]:a_box[3], a_box[0]:a_box[2], :]
        p_img = p_img[p_box[1]:p_box[3], p_box[0]:p_box[2], :]
        n_img = n_img[n_box[1]:n_box[3], n_box[0]:n_box[2], :]

        a_img, p_img, n_img = map(lambda x: cv2.resize(x, (128, 128)), [a_img, p_img, n_img])

        if np.random.rand() < self.aug_color:
            a_img, p_img, n_img = map(self.color_jitter, [a_img, p_img, n_img])
        if np.random.rand() < self.aug_gray:
            a_img, p_img, n_img = map(self.color_gray, [a_img, p_img, n_img])

        if np.random.rand() < self.aug_flip:
            a_img = self.horiz_flip(a_img)
        if np.random.rand() < self.aug_flip:
            p_img = self.horiz_flip(p_img)
        if np.random.rand() < self.aug_flip:
            n_img = self.horiz_flip(n_img)

        if self.debug:
            print(a_box, p_box, n_box)
            self.debug_fn([a_img, p_img, n_img])

        a_img, p_img, n_img = map(self.process, [a_img, p_img, n_img])
        a_img, p_img, n_img = map(lambda x: x.astype(np.float32), [a_img, p_img, n_img])

        return a_img, p_img, n_img

    def debug_fn(self, im):  # [x, y, x, y]
        a_img = im[0]
        p_img = im[1]
        n_img = im[2]

        self.ax1.imshow(a_img)
        self.ax2.imshow(p_img)
        self.ax3.imshow(n_img)
        self.fig.show()
        plt.waitforbuttonpress()

    def process(self, img):  # RGB
        img = cv2.resize(img, (128, 128))

        img = img.astype(np.float32) / 255
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

        return img.transpose(2, 0, 1)


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
