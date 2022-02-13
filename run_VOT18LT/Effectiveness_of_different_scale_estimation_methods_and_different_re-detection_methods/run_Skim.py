# coding=utf-8
import os
import sys
sys.path.append('../code_for_review')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
import numpy as np
from time import time

import config_path as path

from modules.pysot.interface_siamrpn import SiamRPN
from modules.embedder.network import Embedder

from utils import show, save_vot, save_lasot, save_got10k
from modules.pysot.pysot.utils.bbox import get_axis_aligned_bbox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.info('TensorFlow')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.logging.info('TensorFlow')

import keras.backend.tensorflow_backend as KTF


def crop_template_Hao(img, box, times=1.3):
    im_h, im_w, _ = img.shape

    cw = int(box[0] + box[2] / 2)
    ch = int(box[1] + box[3] / 2)

    half_w = int(box[2] / 2 * times)
    half_h = int(box[3] / 2 * times)

    top, bottom, left, right = (0, 0, 0, 0)
    if cw < half_w: left = half_w - cw
    if ch < half_h: top = half_h - ch
    if (cw + half_w) > im_w: right = half_w + cw - im_w
    if (ch + half_h) > im_h: bottom = half_h + ch - im_h

    cw += left
    ch += top

    new_im = cv2.copyMakeBorder(  # BGR [123.68, 116.779, 103.939]
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[123, 117, 104])

    new_im = new_im[
             ch - half_h:ch + half_h,
             cw - half_w:cw + half_w, :]

    return cv2.resize(new_im, (140, 140))

def gen_search_patch_Hao(img, last_reliable_w, last_reliable_h):
    # 2.8 300
    # 2.4 256
    crop_sz = int((last_reliable_w + last_reliable_h) / 2 * 2.4)

    H = int(img.shape[0] / crop_sz) * 256
    W = int(img.shape[1] / crop_sz) * 256
    crop_win = np.array([0, 0, 256, 256], dtype=int)

    if H == 0:
        H = 256
    if W == 0:
        W = 256

    Y, X = np.mgrid[0:H - 128:128, 0:W - 128:128]
    Y = Y.reshape(-1)
    X = X.reshape(-1)

    if len(X) > 500:
        step = int(len(X) / 500)
        '''TypeError: slice indices must be integers or None or have an __index__ method'''
        sel_idx = list(range(len(X)))[::step][:500]

        X = X[sel_idx]
        Y = Y[sel_idx]
    else:
        pass

    search = cv2.resize(img, (W, H))
    search = search.astype(np.float32)[None, :, :, :]
    im = np.ones([len(X), 256, 256, 3])

    pos_i = np.zeros([len(X), 4])
    for i in range(len(X)):
        im[i] = search[0,
                crop_win[1] + Y[i]:crop_win[3] + Y[i],
                crop_win[0] + X[i]:crop_win[2] + X[i], :]

        pos_i[i] = np.array([  # [cx cy w h]
            (crop_win[0] + X[i] + crop_win[2] + X[i]) / 2.0 / W * img.shape[1],
            (crop_win[1] + Y[i] + crop_win[3] + Y[i]) / 2.0 / H * img.shape[0],
            last_reliable_w,
            last_reliable_h,
        ])

    return im, pos_i.astype(int)  # [cx cy w h]


class Tracker(object):

    def __init__(self, image, gt_bbox, vot_flag=True):  # [x y w h]
        bbox = np.array(gt_bbox)

        self.tracker = SiamRPN(cfg_file=path.siam_cfg, snapshot=path.siam_snap)
        self.tracker.init(image, bbox)

        self.center_pos_img = np.array([image.shape[1] / 2, image.shape[0] / 2])
        self.size_1st = self.tracker.tracker.size.copy()

        self.embedder = Embedder().cuda()
        self.embedder.eval()
        self.embedder.init1(image, bbox)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        KTF.set_session(session)
        with tf.keras.utils.CustomObjectScope({
            'relu6': tf.keras.layers.LeakyReLU(0),
            'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}):
            self.branch_z = tf.keras.models.load_model(
                path.skim_z,
                custom_objects={"tf": tf})
            self.branch_search = tf.keras.models.load_model(
                path.skim_s,
                custom_objects={"tf": tf})

        z_im = crop_template_Hao(image, bbox.astype(int))
        z_im = z_im[None, :, :, :].astype(np.float32)
        self.z_feat = self.branch_z.predict(z_im)
        # warm
        self.branch_search.predict([self.z_feat.repeat(1, axis=0), np.random.rand(1, 256, 256, 3)])

        self.vot_flag = vot_flag
        self.show_flag = False
        self.frame = 0

        self.thres_v1 = 1.1
        self.thres_v2 = 0.5

        self.first_w = gt_bbox[2]
        self.first_h = gt_bbox[3]


    def track(self, image, gt):
        self.frame += 1

        box_r, score_r = self.tracker.inference_top_k(image, top_k=20, keep=5)  # [x y x y]
        box_e, score_e, find = self.embedder.inference2(image, box_r, score_r,
                                                        self.frame,
                                                        thres=self.thres_v1)  # [x y x y]
        self.tracker.tracker.center_pos = (box_e[:2] + box_e[2:]) / 2
        self.tracker.tracker.size = box_e[2:] - box_e[:2]

        if find:
            box = box_e
            score = score_e
            if self.embedder.init2_flag is False:
                self.embedder.init2_flag = True
                self.embedder.init2(image, box_e)

        else:
            bk_center_pos = self.tracker.tracker.center_pos.copy()
            bk_size = self.tracker.tracker.size.copy()

            softmax_test_, pos_i = gen_search_patch_Hao(image, self.first_w, self.first_h)
            softmax_test = softmax_test_.astype(np.float32)
            batch_sz = 64

            if softmax_test.shape[0] <= batch_sz:
                kk = softmax_test
                cls_out = self.branch_search.predict([self.z_feat.repeat(kk.shape[0], axis=0), kk]).reshape(-1)

            elif softmax_test.shape[0] > batch_sz:
                cls_out_list = []

                for_i = softmax_test.shape[0] // batch_sz
                for jj in range(for_i):
                    kk = softmax_test[batch_sz * jj:batch_sz * (jj + 1)]
                    cls_out_list.append(
                        self.branch_search.predict([self.z_feat.repeat(kk.shape[0], axis=0), kk]).reshape(-1))

                if softmax_test.shape[0] % batch_sz == 0:
                    pass
                else:
                    kk = softmax_test[batch_sz * (jj + 1):]
                    cls_out_list.append(
                        self.branch_search.predict([self.z_feat.repeat(kk.shape[0], axis=0), kk]).reshape(-1))

                cls_out = np.concatenate(cls_out_list)

            search_rank = np.argsort(-cls_out)
            pos_i = pos_i[search_rank].reshape(-1, 4)

            search_num = 1  # np.minimum(pos_i.shape[0], 3)

            skim_box = []
            skim_score = []
            for s_i in range(search_num):
                self.tracker.tracker.center_pos = pos_i[s_i][:2]  # [cx cy]
                self.tracker.tracker.size = self.size_1st
                box_r, score_r = self.tracker.inference_top_k(image, top_k=20, keep=5)  # [x y x y]
                skim_box.append(box_r.reshape(-1, 4))
                skim_score.append(score_r.reshape(-1))

            box_r = np.concatenate(skim_box, axis=0)
            score_r = np.concatenate(skim_score)

            box_e, score_e, find = self.embedder.inference2(image, box_r, score_r,
                                                            self.frame,
                                                            thres=self.thres_v2)  # [x y x y]

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

    tracker_name = 'vot18lt_Skim'
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
            predict_box, predict_score = tracker.track(image_ori, np.array(gt[frame_i]))
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
