
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import os
import numpy as np
import torch
from modules.pysot.pysot.core.config import cfg
from modules.pysot.pysot.models.model_builder import ModelBuilder
from modules.pysot.pysot.tracker.tracker_builder import build_tracker
from modules.pysot.pysot.utils.model_load import load_pretrain

from utils import overlap_ratio
from nms import nms


class SiamRPN():
    def __init__(self, cfg_file, snapshot):
        # load config
        cfg.merge_from_file(cfg_file)

        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # create model
        self.model = ModelBuilder()
        # load model
        self.model = load_pretrain(self.model, snapshot).cuda().eval()
        # build tracker
        self.tracker = build_tracker(self.model)

    def init(self, img, box):  # [x y w h]
        self.tracker.init(img, box)

    def inference(self, img, thres=0.9):
        outputs = self.tracker.track(img)

        box = np.array(outputs['bbox'])
        score = outputs['best_score']

        if score > thres:
            find = True
        else:
            find = False

        box[2:] = box[2:] + box[:2]  # [x y w h] to [x1 y1 x2 y2]

        box = np.array(box).astype(np.float32).reshape(-1)

        return box, score, find  # [x1 y1 x2 y2] score

    def inference_top_k(self, img, top_k=20, keep=5):  # top K boxes
        outputs = self.tracker.UOF_track_top_k(img, k=top_k)

        box = np.array(outputs['bbox'])
        score = outputs['best_score']

        box[:, 2:] = box[:, 2:] + box[:, :2]  # [x y w h] to [x1 y1 x2 y2]

        keeps, num_to_keep, _ = nms(
            torch.Tensor(box).cuda(),  # [x y w h]
            torch.Tensor(score).cuda(),
            overlap=0.1, top_k=keep)

        num_to_keep = num_to_keep.item()
        keeps = keeps.data.cpu().numpy()
        keeps = keeps[:num_to_keep]

        box = np.array(box[keeps]).astype(np.float32)
        score = np.array(score[keeps]).astype(np.float32)

        return box, score  # [x1 y1 x2 y2] score

    def test_template(self, img, box, video):  # [x y w h]
        imgs = video['img']
        boxes = video['box']  # [x y x y]

        init_box = boxes[0].copy()
        init_box[2:] = init_box[2:] - init_box[:2]  # [x y w h]

        bbox = box.copy()

        bk_zf = [self.model.zf[0].clone().detach(),
                 self.model.zf[1].clone().detach(),
                 self.model.zf[2].clone().detach()]
        bk_center_pos = self.tracker.center_pos.copy()
        bk_size = self.tracker.size.copy()

        # get test template
        self.tracker.init(img, bbox)

        iou_list = []
        for t_id, t_img in enumerate(imgs):  # [x y x y]
            self.tracker.center_pos = np.array([127, 127])
            self.tracker.size = np.array([init_box[2], init_box[3]])

            box_r, _, _ = self.inference(t_img)  # [x y x y]
            iou_list.append(overlap_ratio(box_r, boxes[t_id]))

        self.model.zf[0] = bk_zf[0]
        self.model.zf[1] = bk_zf[1]
        self.model.zf[2] = bk_zf[2]
        self.tracker.center_pos = bk_center_pos
        self.tracker.size = bk_size

        return np.mean(iou_list)