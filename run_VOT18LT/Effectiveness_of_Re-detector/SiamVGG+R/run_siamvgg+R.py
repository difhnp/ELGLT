
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import numpy as np
import cv2
from time import time
from src.siamvggtracker import SiamVGGTracker
from utils import show, save_vot, save_lasot, save_got10k
# from modules.pysot.toolkit.utils.region import vot_overlap, vot_float2str


ltb_path = '/data1/Dataset/VOT/LTB35'
save_path = './results'
tracker_name = 'SiamVGG+R'


vid_name = os.listdir(ltb_path)
vid_name.remove('list.txt')
vid_name.remove('VOT2018-LT.json')
vid_name.sort()


SAVE_FLAG = True

THRES = 3.0*1e-5

# for vid_id in [vid_name.index('dragon')]:
for vid_id in range(0, 35):

    print('processing...video', vid_id, vid_name[vid_id])

    """ groundtruth"""
    gt = np.loadtxt(os.path.join(ltb_path, vid_name[vid_id], 'groundtruth.txt'), delimiter=',')

    """init"""
    init_path = os.path.join(ltb_path, vid_name[vid_id], 'color', '{:0>8d}.jpg'.format(0 + 1))
    init_img = cv2.imread(init_path)

    tracker = SiamVGGTracker(init_img, gt[0], thres=THRES)

    box_list = []
    score_list = []
    time_list = []
    fps_list = []
    box_list.append('1\n')


    for frame_i in range(1, len(gt)):  # 1122
        print('processing...video', vid_id, vid_name[vid_id], '%d / %d' % (frame_i + 1, len(gt)))
        im_path = os.path.join(ltb_path, vid_name[vid_id], 'color', '{:0>8d}.jpg'.format(frame_i + 1))
        image_ori = cv2.imread(im_path)

        tic = time()
        region, confidence = tracker.track(image_ori)
        toc = time()


        # region = region.astype(int)
        # im = cv2.rectangle(image_ori, (region[0], region[1]), (region[2]+region[0], region[3]+region[1]), (0,255,0), 2)
        # img = np.array(im) # BGR
        # cv2.imshow(tracker_name, img)
        # cv2.waitKey(1)


        fps_list.append(toc - tic)
        time_list.append('{:.6f}\n'.format(toc - tic))
        score_list.append('{:.6f}\n'.format(confidence))

        box_list.append('{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(region[0], region[1], region[2], region[3]))

        # print('{:0>2d}{:>14s} speed: {:6.2f} fps'.format(vid_id, vid_name[vid_id], (len(gt)-1) / np.sum(fps_list)))

    save_vot(
        vid_name[vid_id], tracker_name=tracker_name, save_path='./results',
        box_list=box_list, confidence_list=score_list, time_list=time_list, tag='longterm'
    )
