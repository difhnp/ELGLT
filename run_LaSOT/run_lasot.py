# coding=utf-8
import os
import sys
sys.path.append('../code_for_review')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import cv2
import numpy as np
from time import time

import config_path as path

from modules.pysot.interface_siamrpn import SiamRPN
from modules.embedder.network import Embedder
from modules.guider.network import Guider

from utils import show, save_vot, save_lasot, save_got10k
from modules.pysot.pysot.utils.bbox import get_axis_aligned_bbox

class Tracker(object):

    def __init__(self, image, bbox, vot_flag=True):  # [x y w h]
        bbox = np.array(bbox)

        self.tracker = SiamRPN(cfg_file=path.siam_cfg, snapshot=path.siam_snap)
        self.tracker.init(image, bbox)

        self.center_pos_img = np.array([image.shape[1] / 2, image.shape[0] / 2])
        self.size_1st = self.tracker.tracker.size.copy()

        self.guider = Guider().cuda()
        self.guider.eval()
        self.guider.init(image, bbox)

        self.embedder = Embedder().cuda()
        self.embedder.eval()
        self.embedder.init1(image, bbox)

        self.vot_flag = vot_flag
        self.show_flag = False
        self.frame = 0

        self.thres_v1 = 1.3
        self.thres_v2 = 0.4

        self.thres_map = 0.2
        self.thres_bin = 0.4
        self.w_map = 0.5
        self.thres_area = 100

    def track(self, image, gt):
        self.frame += 1

        box_r, score_r = self.tracker.inference_top_k(image, top_k=20, keep=5)  # [x y x y]
        box_e, score_e, find = self.embedder.inference2(image, box_r, score_r,
                                                        self.frame, thres=self.thres_v1)  # [x y x y]
        self.tracker.tracker.center_pos = (box_e[:2] + box_e[2:]) / 2
        self.tracker.tracker.size = box_e[2:] - box_e[:2]

        if find:
            box = box_e
            score = score_e

            if self.embedder.init2_flag is False:
                self.embedder.init2_flag = True
                self.embedder.init2(image, box_e)
        else:
            obj_map = self.guider.inference(image)  # [x y x y]

            bk_center_pos = self.tracker.tracker.center_pos.copy()
            bk_size = self.tracker.tracker.size.copy()

            if obj_map.max() < self.thres_map:
                self.tracker.tracker.center_pos = self.center_pos_img
                self.tracker.tracker.size = self.size_1st
                box_r, score_r, find = self.tracker.inference(image)
            else:
                # find peak
                obj_w, obj_h = np.where(obj_map == obj_map.max())
                obj_w = obj_w[0]
                obj_h = obj_h[0]

                obj_map[obj_map > self.thres_bin] = 1
                obj_map[obj_map <= self.thres_bin] = 0
                contours, _ = cv2.findContours(obj_map.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                cnt_area = [cv2.contourArea(cnt) for cnt in contours]
                if len(contours) != 0 and np.max(cnt_area) > self.thres_area:
                    contour = contours[np.argmax(cnt_area)]
                    x, y, w, h = cv2.boundingRect(contour)
                    side = np.sqrt(w * h)
                    self.tracker.tracker.center_pos = np.array([x + w / 2, y + h / 2])
                    self.tracker.tracker.size = self.size_1st * (1-self.w_map) + np.array([side, side]) * self.w_map
                else:  # empty mask
                    self.tracker.tracker.center_pos = np.array([obj_w, obj_h])
                    self.tracker.tracker.size = self.size_1st

                box_r, score_r = self.tracker.inference_top_k(image, top_k=20, keep=5)  # [x y x y]
                box_e, score_e, find = self.embedder.inference2(image, box_r, score_r,
                                                                self.frame, thres=self.thres_v2)  # [x y x y]
            if find:
                box = box_e
                score = score_e
            else:
                box = box_e
                score = score_e
                self.tracker.tracker.center_pos = bk_center_pos
                self.tracker.tracker.size = bk_size

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
    gt_box = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

    image_file = handle.frame()
    if not image_file:
        sys.exit(0)

    image = cv2.imread(image_file)
    tracker = Tracker(image, gt_box, vot_flag=VOT_FLAG)

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
    from modules.pysot.toolkit.utils.region import vot_float2str

    SAVE_FLAG = True

    tracker_name = 'lasot_Ours'
    print('===========================================================================================================')
    print('Tracker:', tracker_name)

    data_name = 'LaSOT'
    data_dir = path.lasot

    data_set = DatasetFactory.create_dataset(name=data_name, dataset_root=data_dir, load_img=False)

    fps_sum = []
    for v_idx, video in enumerate(data_set):
        box_list = []
        score_list = []
        time_list = []
        fps_list = []

        title = video.name

        if v_idx < 0:
            continue
        if os.path.exists(os.path.join('./results', tracker_name, '{:s}.txt'.format(title))):
            continue

        for frame_i, (img, gt_bbox) in enumerate(video):

            if frame_i == 0:

                tic = time()
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_box = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

                tracker = Tracker(img, gt_box, vot_flag=VOT_FLAG)
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
            else:
                tic = time()
                predict_box, predict_score = tracker.track(img, np.array(gt_bbox))
                toc = time()

                fps_list.append(toc - tic)
                time_list.append('{:.6f}\n'.format(toc - tic))
                score_list.append('{:.6f}\n'.format(predict_score))

                box_list.append('{},{},{},{}\n'.format(
                    vot_float2str("%.4f", predict_box[0]),
                    vot_float2str("%.4f", predict_box[1]),
                    vot_float2str("%.4f", predict_box[2]),
                    vot_float2str("%.4f", predict_box[3])))

        fps_sum.append(1 / np.mean(fps_list))
        print('{:0>2d} {:<20s} speed: {:6.2f} fps'.format(v_idx, title, 1 / np.mean(fps_list)))
        if SAVE_FLAG:
            if data_name == 'VOT2018-LT':
                save_vot(
                    title, tracker_name=tracker_name, save_path='./results',
                    box_list=box_list, confidence_list=score_list, time_list=time_list, tag='longterm'
                )
            elif data_name == 'VOT2018':
                save_vot(
                    title, tracker_name=tracker_name, save_path='./results',
                    box_list=box_list, tag='unsupervised'
                )
            elif data_name == 'GOT-10k':
                save_got10k(
                    title, tracker_name=tracker_name, save_path='./results',
                    box_list=box_list, time_list=time_list
                )
            else:  # LaSOT
                save_lasot(
                    title, tracker_name=tracker_name, save_path='./results', box_list=box_list
                )
    print('{:0>2d} {:<20s} speed: {:6.2f} fps'.format(99, '', np.mean(fps_sum)))
    print('===========================================================================================================')
