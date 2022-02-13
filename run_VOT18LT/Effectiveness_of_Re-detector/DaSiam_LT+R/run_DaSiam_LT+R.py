import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

sys.path.append('/home/sensetime/vot2018/2018/final/vot-toolkit/tracker/DaSiamRPN')
from vot2018_longterm import *

from modules.pysot.pysot.utils.bbox import get_axis_aligned_bbox
from modules.pysot.toolkit.datasets import DatasetFactory
from modules.pysot.toolkit.utils.region import vot_overlap, vot_float2str
from utils2 import show, save_vot, save_lasot, save_got10k
from modules.guider.network import Guider

from time import time

class Struct:
    def __init__(self, kwargs):
        self.__dict__.update(kwargs)


def config_params(vot=False):
    p = {}

    if vot:
        p['window_influence'] = 0.28
        p['lr']  = 0.22
        p['scale_penalty_k'] = 0.11
        p['confidence_low'] = 0.74
        p['confidence_high'] = 0.996

        p['c_sz'] = 8
        p['cur_avgChans'] = False
        p['model'] = 'alex_175'
        p['norm'] = False
        p['scale_num'] = 1
    p = Struct(p)
    return p


p = config_params(vot=True)

sp = SizePenalty(p.scale_penalty_k)

#=========================================================
data_name = 'VOT2018-LT'
data_dir = '/data1/Dataset/VOT/LTB35/'

data_set = DatasetFactory.create_dataset(name=data_name, dataset_root=data_dir, load_img=False)

SAVE_FLAG = True

for thres_map in [0.31]:
    thres = 0.7
    thres2 = 0.65

    model_name = 'DaSiam_LT+R'
    for v_idx, video in enumerate(data_set):
        title = video.name

        if v_idx < 0:
            continue

        box_list = []
        score_list = []
        time_list = []
        fps_list = []

        for img_idx, (im, gt_bbox) in enumerate(video):
            # print img_idx
            if img_idx == 0:
                tic = time()
                net = models.models[p.model]()
                net.load_model()
                net = net.cuda()
                rpn = net.generate_rpn()

                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_box = [cx - (w - 1) / 2.0, cy - (h - 1) / 2.0, w, h]
                x1, y1, w, h = gt_box
                rect = [x1, y1, x1+w, y1+h]

                top_k = 1
                long_term = False

                tracker = SiamRPN(im, rect, net, rpn, vot=True, p=p)

                guider = Guider().cuda()
                guider.eval()
                guider.init(im, np.array(gt_box))
                # thres_map = 0.2
                thres_bin = 0.4
                w_map = 0.2
                thres_area = 100

                center_pos_img = np.array([im.shape[0] / 2, im.shape[1] / 2])  # cy cx
                size_1st = np.array(tracker.targetSize)

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
                rect, confidence = tracker.track(im, long_term=False)

                if confidence > thres:
                    x1, y1, x2, y2 = rect
                    rect = np.array([x1, y1, x2 - x1, y2 - y1])

                    box = rect
                    score = confidence
                else:
                    obj_map = guider.inference(im)  # [x y x y]

                    bk_center_pos = np.array(tracker.targetPosition)  # cy cx
                    bk_size = np.array(tracker.targetSize)  # h w

                    if obj_map.max() < thres_map:
                        tracker.targetPosition = center_pos_img.astype(np.float32)
                        tracker.targetSize = size_1st.astype(np.float32)
                        rect, confidence = tracker.track(im, long_term=False)
                    else:
                        # find peak
                        obj_w, obj_h = np.where(obj_map == obj_map.max())
                        obj_w = obj_w[0]
                        obj_h = obj_h[0]

                        obj_map[obj_map > thres_bin] = 1
                        obj_map[obj_map <= thres_bin] = 0
                        contours, _ = cv2.findContours(obj_map.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
                        if len(contours) != 0 and np.max(cnt_area) > thres_area:
                            contour = contours[np.argmax(cnt_area)]
                            x, y, w, h = cv2.boundingRect(contour)
                            side = np.sqrt(w * h)

                            tracker.targetPosition = np.array([y + h / 2.0, x + w / 2.0], dtype=np.float32)  # cy cx
                            tracker.targetSize = size_1st * (1 - w_map) + np.array(
                                [side, side], dtype=np.float32) * w_map
                        else:  # empty mask
                            tracker.targetPosition = np.array([obj_h, obj_w], dtype=np.float32)
                            tracker.targetSize = np.array(size_1st, dtype=np.float32)

                        rect, confidence = tracker.track(im, long_term=False)


                    if confidence > thres2:
                        x1, y1, x2, y2 = rect
                        rect = np.array([x1, y1, x2 - x1, y2 - y1])
                        box = rect
                        score = confidence
                    else:
                        x1, y1, x2, y2 = rect
                        rect = np.array([x1, y1, x2 - x1, y2 - y1])
                        box = rect
                        score = confidence
                        tracker.targetPosition = bk_center_pos
                        tracker.targetSize = bk_size

                toc = time()

                # if score < p.confidence_low:
                #     long_term = True
                # elif score > p.confidence_high:
                #     long_term = False

                fps_list.append(toc - tic)
                time_list.append('{:.6f}\n'.format(toc - tic))
                score_list.append('{:.6f}\n'.format(score))

                box_list.append('{},{},{},{}\n'.format(
                    vot_float2str("%.4f", box[0]),
                    vot_float2str("%.4f", box[1]),
                    vot_float2str("%.4f", box[2]),
                    vot_float2str("%.4f", box[3])))

                # show(im, 0, [box[0], box[1], box[0]+box[2], box[1]+box[3]], confidence)

        print('{:0>2d}{:>14s} speed: {:6.2f} fps {}'.format(v_idx, title, img_idx/np.sum(fps_list), img_idx))
        if SAVE_FLAG:
            if data_name == 'VOT2018-LT':
                save_vot(
                    title, tracker_name=model_name, save_path='./results',
                    box_list=box_list, confidence_list=score_list, time_list=time_list, tag='longterm'
                )
