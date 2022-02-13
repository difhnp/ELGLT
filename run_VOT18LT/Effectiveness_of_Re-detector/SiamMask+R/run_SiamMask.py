# coding=utf-8
import os
import sys
import cv2

import random
import numpy as np
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn.functional as F

import config_path as path

from modules.pysot.interface_siammask import SiamMask
from modules.pysot.pysot.utils.bbox import get_axis_aligned_bbox

from modules.guider.network import Guider


from utils import show, save_vot, save_lasot, save_got10k


# 0.632
class Tracker(object):

    def __init__(self, image, bbox, vot_flag=True):  # [x y w h]
        bbox = np.array(bbox)

        # print('=> init RPN')
        self.tracker = SiamMask(cfg_file=path.siammask_cfg, snapshot=path.siammask_snap)
        self.tracker.init(image, bbox)

        self.center_pos_img = np.array([image.shape[1] / 2, image.shape[0] / 2])
        self.size_1st = self.tracker.tracker.size.copy()

        # print('=> init Guider')
        self.guider = Guider().cuda()
        self.guider.eval()
        self.guider.init(image, bbox)


        self.vot_flag = vot_flag
        self.show_flag = False
        self.frame = 0


        self.thres_map = 0.2
        self.thres_bin = 0.4
        self.w_map = 0.4
        self.thres_area = 100


    def track(self, image, gt):
        self.frame += 1

        # print('-------------------------------------')
        tic_tmp = time()
        box_e, _, score_e, find = self.tracker.inference(image)  # [x y x y]
        # box_r, score_r = self.tracker.inference_top_k(image, top_k=20, keep=5)  # [x y x y]
        # box_e, score_e, find = self.embedder.inference3(image, box_r, score_r,
        #                                                 self.frame,
        #                                                 thres=self.thres_v1,
        #                                                 alpha=self.alpha,
        #                                                 alpha2=self.alpha2)  # [x y x y]
        # self.tracker.tracker.center_pos = (box_e[:2] + box_e[2:]) / 2
        # self.tracker.tracker.size = box_e[2:] - box_e[:2]

        box = box_e
        score = score_e

        if find:
            box = box_e
            score = score_e


        else:
            self.tracker.tracker.center_pos = self.center_pos_img
            self.tracker.tracker.size = self.size_1st
            box_e, _, score_e, find = self.tracker.inference(image)  # [x y x y]
            box = box_e
            score = score_e

        #     # print('-------------------------------------')
        #     tic_tmp = time()
        #     obj_map = self.guider.inference(image)  # [x y x y]
        #     # print('guider  : {:.2f} fps'.format(1 / (time() - tic_tmp)))
        #
        #     bk_center_pos = self.tracker.tracker.center_pos.copy()
        #     bk_size = self.tracker.tracker.size.copy()
        #
        #     if obj_map.max() < self.thres_map:
        #         self.tracker.tracker.center_pos = self.center_pos_img
        #         self.tracker.tracker.size = self.size_1st
        #         box_r, score_r, find = self.tracker.inference(image)
        #     else:
        #         # find peak
        #         obj_w, obj_h = np.where(obj_map == obj_map.max())
        #         obj_w = obj_w[0]
        #         obj_h = obj_h[0]
        #
        #         obj_map[obj_map > self.thres_bin] = 1
        #         obj_map[obj_map <= self.thres_bin] = 0
        #         contours, _ = cv2.findContours(obj_map.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        #         cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        #         if len(contours) != 0 and np.max(cnt_area) > self.thres_area:
        #             contour = contours[np.argmax(cnt_area)]
        #             x, y, w, h = cv2.boundingRect(contour)
        #             side = np.sqrt(w * h)
        #             self.tracker.tracker.center_pos = np.array([x + w / 2, y + h / 2])
        #             self.tracker.tracker.size = self.size_1st * (1 - self.w_map) + np.array([side, side]) * self.w_map
        #         else:  # empty mask
        #             self.tracker.tracker.center_pos = np.array([obj_w, obj_h])
        #             self.tracker.tracker.size = self.size_1st
        #
        #         # box_r, score_r, find = self.tracker.inference(image)  # [x y x y]
        #         box_r, score_r = self.tracker.inference_top_k(image, top_k=20, keep=5)  # [x y x y]
        #         box_e, score_e, find = self.embedder.inference3(image, box_r, score_r,
        #                                                         self.frame,
        #                                                         thres=self.thres_v2,
        #                                                         alpha=self.alpha,
        #                                                         alpha2=self.alpha2)  # [x y x y]
        #
        #         # obj_map = obj_map * 255
        #         # image = image.astype(np.float32)
        #         # image[:, :, 2] = image[:, :, 2] * 0.1 + obj_map * 0.9
        #         # image = image.astype(np.uint8)
        #
        #     if find:
        #         box = box_e
        #         score = score_e
        #     else:
        #         box = box_e
        #         score = score_e
        #         self.tracker.tracker.center_pos = bk_center_pos
        #         self.tracker.tracker.size = bk_size

        if self.show_flag:
            if len(gt) < 4:
                show(image, self.frame, box, score)
            else:
                show(image, self.frame, box, score, gt_box=gt)

        if self.vot_flag:
            return vot.Rectangle(box[0], box[1], box[2] - box[0], box[3] - box[1]), score
        else:
            return np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]]), score


# ==================================================================================================================== #


VOT_FLAG = False

if VOT_FLAG:
    import vot

    handle = vot.VOT("polygon")
    region = handle.region()
    try:
        region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                           region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])
    except AttributeError:
        region = np.array(region)

    cx, cy, w, h = get_axis_aligned_bbox(region)
    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

    image_file = handle.frame()
    if not image_file:
        sys.exit(0)

    image = cv2.imread(image_file)
    tracker = Tracker(image, gt_bbox_)

    while True:
        image_file = handle.frame()
        if not image_file:
            break
        image = cv2.imread(image_file)
        region, confidence = tracker.track(image)
        handle.report(region, confidence)

# ==================================================================================================================== #

else:

    from modules.pysot.toolkit.datasets import DatasetFactory
    from modules.pysot.toolkit.utils.region import vot_overlap, vot_float2str


    model_name = 'SiamMask'

    ltb_path = '/data1/Dataset/VOT/LTB35'
    vid_name = os.listdir(ltb_path)
    vid_name.remove('list.txt')
    vid_name.remove('VOT2018-LT.json')
    vid_name.sort()

    data_name = 'VOT2018-LT'
    data_dir = '/data1/Dataset/VOT/LTB35/'

    SAVE_FLAG = True

    for vid_id in range(4, 35):

        title = vid_name[vid_id]
        # print('processing...video', vid_id, title)

        """ groundtruth"""
        gt = np.loadtxt(os.path.join(ltb_path, title, 'groundtruth.txt'), delimiter=',')

        """init"""
        init_path = os.path.join(ltb_path, title, 'color', '{:0>8d}.jpg'.format(0 + 1))
        init_img = cv2.imread(init_path)

        box_list = []
        score_list = []
        time_list = []
        fps_list = []

        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt[0]))
        gt_box = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

        tic = time()
        tracker = Tracker(init_img, gt_box, vot_flag=VOT_FLAG)
        toc = time()

        fps_list.append(toc - tic)
        time_list.append('{:.6f}\n'.format(toc - tic))

        if data_name == 'VOT2018-LT':
            box_list.append('1\n')
        else:
            box_list.append('{},{},{},{}\n'.format(
                vot_float2str("%.4f", gt_box[0]),
                vot_float2str("%.4f", gt_box[1]),
                vot_float2str("%.4f", gt_box[2]),
                vot_float2str("%.4f", gt_box[3])))

        for frame_i in range(1, len(gt)):  # 1122
            # print('processing...video', vid_id, title, '%d / %d' % (frame_i + 1, len(gt)))
            im_path = os.path.join(ltb_path, title, 'color', '{:0>8d}.jpg'.format(frame_i + 1))
            image_ori = cv2.imread(im_path)

            tic = time()
            predict_box, predict_score = tracker.track(image_ori, np.array(gt[frame_i]))
            toc = time()

            # print('-------------------------------------')
            # print('total   : {:.2f} fps'.format(1 / (toc - tic)))

            fps_list.append(toc - tic)
            time_list.append('{:.6f}\n'.format(toc - tic))
            score_list.append('{:.6f}\n'.format(predict_score))

            box_list.append('{},{},{},{}\n'.format(
                vot_float2str("%.4f", predict_box[0]),
                vot_float2str("%.4f", predict_box[1]),
                vot_float2str("%.4f", predict_box[2]),
                vot_float2str("%.4f", predict_box[3])))

        print('{:0>2d}{:>14s} speed: {:6.2f} fps'.format(vid_id, title, frame_i / (np.sum(fps_list) - 1)))
        if SAVE_FLAG:
            if data_name == 'VOT2018-LT':
                save_vot(
                    title, tracker_name=model_name, save_path='./results',
                    box_list=box_list, confidence_list=score_list, time_list=time_list, tag='longterm'
                )
            elif data_name == 'VOT2018':
                save_vot(
                    title, tracker_name=model_name, save_path='./results',
                    box_list=box_list, tag='unsupervised'
                )
            elif data_name == 'GOT-10k':
                save_got10k(
                    title, tracker_name=model_name, save_path='./results',
                    box_list=box_list, time_list=time_list
                )
            else:  # LaSOT
                save_lasot(
                    title, tracker_name=model_name, save_path='./results', box_list=box_list
                )
