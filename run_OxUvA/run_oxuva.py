# coding=utf-8
import os
import sys
sys.path.append('../code_for_review')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

import oxuva as oxuva
import argparse


def rect_from_opencv(rect, imsize_hw):
    imheight, imwidth = imsize_hw
    xmin_abs, ymin_abs, width_abs, height_abs = rect
    xmax_abs = xmin_abs + width_abs
    ymax_abs = ymin_abs + height_abs
    return {
        'xmin': xmin_abs / imwidth,
        'ymin': ymin_abs / imheight,
        'xmax': xmax_abs / imwidth,
        'ymax': ymax_abs / imheight,
    }

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

        self.thres_v = 1.1
        self.thres_r = 0.97

    def track(self, image):
        self.frame += 1

        box_r, score_r = self.tracker.inference_top_k(image, top_k=20, keep=5)  # [x y x y]
        box_e, score_e, score_v, find = self.embedder.inference2ox(image, box_r, score_r,
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
                box_e, score_e, score_v, find = self.embedder.inference2ox(image, box_r, score_r,
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
            show(image, self.frame, box, score)

        if score_v < self.thres_v and score_e > self.thres_r:
            present = True
        else:
            present = False

        rect = rect_from_opencv(np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]]), imsize_hw=(image.shape[0], image.shape[1]))
        return oxuva.make_prediction(present=present, score=score_e, **rect)

# ==================================================================================================================== #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=path.oxuva)
    parser.add_argument('--predictions_dir', default='./results')
    parser.add_argument('--data', default='test')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--tracker', default='Ours')
    global args
    args = parser.parse_args()

    print('===========================================================================================================')
    print('Tracker:', 'Ours')


    tracker_id = 'cv' + args.tracker
    tracker_preds_dir = os.path.join(args.predictions_dir, args.data, tracker_id)
    if not os.path.exists(tracker_preds_dir):
        os.makedirs(tracker_preds_dir)

    tasks_file = os.path.join(args.data_dir, 'tasks', args.data + '.csv')
    with open(tasks_file, 'r') as fp:
        tasks = oxuva.load_dataset_tasks_csv(fp)

    imfile = lambda vid, t: os.path.join(
        args.data_dir, 'images', args.data, vid, '{:06d}.jpeg'.format(t))

    """tracker"""
    for key, task in tasks.items():
        vid, obj = key
        if args.verbose:
            print(vid, obj)
        preds_file = os.path.join(tracker_preds_dir, '{}_{}.csv'.format(vid, obj))
        if os.path.exists(preds_file):
            continue

        preds = oxuva.SparseTimeSeries()

        image = cv2.imread(imfile(vid, task.init_time))

        selection = np.array([
            task.init_rect['xmin'],
            task.init_rect['xmax'],
            task.init_rect['ymin'],
            task.init_rect['ymax']]) # xmin xmax ymin ymax

        selection[:2] *= image.shape[1]
        selection[2:] *= image.shape[0]
        init_box = np.array([selection[0], selection[2],
                                selection[1]-selection[0],
                                selection[3]-selection[2]])  # x y w h

        print('tracking...', vid)
        tracker = Tracker(image, init_box, vot_flag=False)

        start = time.time()
        for t in range(task.init_time + 1, task.last_time + 1):
            image = cv2.imread(imfile(vid, t))

            preds[t] = tracker.track(image)
        dur_sec = time.time() - start

        if args.verbose:
            fps = (task.last_time - task.init_time + 1) / dur_sec
            print('fps {:.3g}'.format(fps))

        tmp_preds_file = os.path.join(tracker_preds_dir, '{}_{}.csv.tmp'.format(vid, obj))
        with open(tmp_preds_file, 'w') as fp:
            oxuva.dump_predictions_csv(vid, obj, preds, fp)
        os.rename(tmp_preds_file, preds_file)

    print('===========================================================================================================')


if __name__ == '__main__':
    import time
    main()
