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


def get_box(im, pos, size, model_sz=255):
    """
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
        avg_chans: channel average
    """
    w_z = size[0] + 0.5 * np.sum(size)
    h_z = size[1] + 0.5 * np.sum(size)
    s_z = np.sqrt(w_z * h_z)

    s_x = s_z * (255 / 127)
    original_sz = round(s_x)

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

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    bbbox = np.array([context_xmin, context_ymin, context_xmax, context_ymax])
    bbbox[0::2] = np.clip(bbbox[0::2], 0, im.shape[1])
    bbbox[1::2] = np.clip(bbbox[1::2], 0, im.shape[0])

    return bbbox


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

        self.thres_v1 = 1.1
        self.thres_v2 = 0.5

        self.thres_map = 0.2
        self.thres_bin = 0.4
        self.w_map = 0.4
        self.thres_area = 100

    def track(self, image, gt, title=None):
        self.frame += 1

        gt_show = gt.copy().astype(int)

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

                    # if not np.isnan(gt[1]):
                    #     tmp_im = image.copy().astype(np.float32)
                    #     obj_map = obj_map * 255
                    #     tmp_im[:, :, 2] = tmp_im[:, :, 2] * 0.1 + obj_map * 0.9
                    #     tmp_im = tmp_im.astype(np.uint8)
                    #
                    #     if not os.path.exists(os.path.join(path.project_path, 'fig_obj/{:s}'.format(title))):
                    #         os.mkdir(os.path.join(path.project_path, 'fig_obj/{:s}'.format(title)))
                    #     save_dir = os.path.join(path.project_path, 'fig_obj/{:s}/{:0>6d}.jpg'.format(title, self.frame))
                    #     cv2.imwrite(save_dir, tmp_im)

                else:  # empty mask
                    self.tracker.tracker.center_pos = np.array([obj_w, obj_h])
                    self.tracker.tracker.size = self.size_1st

                box_r, score_r = self.tracker.inference_top_k(image, top_k=20, keep=5)  # [x y x y]
                box_e, score_e, find = self.embedder.inference2(image, box_r, score_r,
                                                                self.frame, thres=self.thres_v2)  # [x y x y]

            # if not np.isnan(gt[1]):
            #     tmp_box = np.array([self.tracker.tracker.center_pos[0], self.tracker.tracker.center_pos[1],
            #                         self.tracker.tracker.size[0], self.tracker.tracker.size[1]])  # [cx cy w h]
            #     tmp_box[:2] = tmp_box[:2] - tmp_box[2:] * 0.5
            #     tmp_box[2:] = tmp_box[2:] + tmp_box[:2]
            #     tmp_box = tmp_box.astype(int)
            #     tmp_im2 = image.copy()
            #     cv2.rectangle(tmp_im2, (tmp_box[0], tmp_box[1]), (tmp_box[2], tmp_box[3]), (0, 255, 0), 2)
            #     cv2.rectangle(tmp_im2, (gt_show[0], gt_show[1]), (gt_show[0]+gt_show[2], gt_show[1]+gt_show[3]), (0, 0, 255), 2)
            #
            #     if not os.path.exists(os.path.join(path.project_path, 'fig_ref_box/{:s}'.format(title))):
            #         os.mkdir(os.path.join(path.project_path, 'fig_ref_box/{:s}'.format(title)))
            #     save_dir2 = os.path.join(path.project_path, 'fig_ref_box/{:s}/{:0>6d}.jpg'.format(title, self.frame))
            #     cv2.imwrite(save_dir2, tmp_im2)
            #
            #     tmp_box = get_box(image, self.tracker.tracker.center_pos, self.tracker.tracker.size)
            #     tmp_box = tmp_box.astype(int)
            #     tmp_im3 = image.copy()
            #     cv2.rectangle(tmp_im3, (tmp_box[0], tmp_box[1]), (tmp_box[2], tmp_box[3]), (0, 255, 0), 2)
            #     cv2.rectangle(tmp_im3, (gt_show[0], gt_show[1]), (gt_show[0]+gt_show[2], gt_show[1]+gt_show[3]), (0, 0, 255), 2)
            #
            #     if not os.path.exists(os.path.join(path.project_path, 'fig_box/{:s}'.format(title))):
            #         os.mkdir(os.path.join(path.project_path, 'fig_box/{:s}'.format(title)))
            #     save_dir3 = os.path.join(path.project_path, 'fig_box/{:s}/{:0>6d}.jpg'.format(title, self.frame))
            #     cv2.imwrite(save_dir3, tmp_im3)

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
    from modules.pysot.toolkit.utils.region import vot_float2str

    SAVE_FLAG = True

    tracker_name = 'vot18lt_Ours'
    print('===========================================================================================================')
    print('Tracker:', tracker_name)

    data_name = 'VOT2018-LT'
    data_dir = path.vot18lt

    vid_name = os.listdir(data_dir)
    vid_name.remove('list.txt')
    vid_name.remove('VOT2018-LT.json')
    vid_name.sort()

    fps_sum = []
    for vid_id in range(0, 35):
        box_list = []
        score_list = []
        time_list = []
        fps_list = []

        title = vid_name[vid_id]

        """ groundtruth"""
        gt = np.loadtxt(os.path.join(data_dir, title, 'groundtruth.txt'), delimiter=',')

        """init"""
        init_path = os.path.join(data_dir, title, 'color', '{:0>8d}.jpg'.format(0 + 1))
        init_img = cv2.imread(init_path)

        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt[0]))
        gt_box = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]

        tic = time()
        tracker = Tracker(init_img, gt_box, vot_flag=VOT_FLAG)
        toc = time()

        fps_list.append(toc - tic)
        time_list.append('{:.6f}\n'.format(toc - tic))

        if data_name == 'VOT2018-LT':
            box_list.append('1\n')
        elif data_name == 'TLP':
            box_list.append('{} {} {} {}\n'.format(
                vot_float2str("%.4f", gt_box[0]),
                vot_float2str("%.4f", gt_box[1]),
                vot_float2str("%.4f", gt_box[2]),
                vot_float2str("%.4f", gt_box[3])))
        else:
            box_list.append('{},{},{},{}\n'.format(
                vot_float2str("%.4f", gt_box[0]),
                vot_float2str("%.4f", gt_box[1]),
                vot_float2str("%.4f", gt_box[2]),
                vot_float2str("%.4f", gt_box[3])))

        for frame_i in range(1, len(gt)):  # 1122
            # print('processing...video', vid_id, title, '%d / %d' % (frame_i + 1, len(gt)))
            im_path = os.path.join(data_dir, title, 'color', '{:0>8d}.jpg'.format(frame_i + 1))
            image_ori = cv2.imread(im_path)

            tic = time()
            predict_box, predict_score = tracker.track(image_ori, np.array(gt[frame_i]), title=title)
            toc = time()

            fps_list.append(toc - tic)
            time_list.append('{:.6f}\n'.format(toc - tic))
            score_list.append('{:.6f}\n'.format(predict_score))

            if data_name == 'TLP':
                box_list.append('{} {} {} {}\n'.format(
                    vot_float2str("%.4f", gt_box[0]),
                    vot_float2str("%.4f", gt_box[1]),
                    vot_float2str("%.4f", gt_box[2]),
                    vot_float2str("%.4f", gt_box[3])))
            else:
                box_list.append('{},{},{},{}\n'.format(
                    vot_float2str("%.4f", predict_box[0]),
                    vot_float2str("%.4f", predict_box[1]),
                    vot_float2str("%.4f", predict_box[2]),
                    vot_float2str("%.4f", predict_box[3])))

        fps_sum.append(1 / np.mean(fps_list))
        print('{:0>2d} {:<20s} speed: {:6.2f} fps'.format(vid_id, title, 1 / np.mean(fps_list)))
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
