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

from utils import show, save_vot, save_lasot, save_got10k
from modules.pysot.pysot.utils.bbox import get_axis_aligned_bbox

class Tracker(object):

    def __init__(self, image, bbox, vot_flag=True):  # [x y w h]
        bbox = np.array(bbox)

        self.tracker = SiamRPN(cfg_file=path.siam_cfg, snapshot=path.siam_snap)
        self.tracker.init(image, bbox)

        self.vot_flag = vot_flag
        self.show_flag = False
        self.frame = 0

    def track(self, image, gt):
        self.frame += 1
        box_r, score_r, find = self.tracker.inference(image)  # [x y x y]
        box = box_r.reshape(-1)
        score = score_r

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

    tracker_name = 'lasot_Base'
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

        fps_sum.append(frame_i / np.sum(fps_list))
        print('{:0>2d}{:>14s} speed: {:6.2f} fps'.format(v_idx, title, frame_i / np.sum(fps_list)))
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
