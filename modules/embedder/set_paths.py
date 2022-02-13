import sys
import socket
from os.path import join as osjoin

project_dir = '/data2/Documents/Workspce/PAMI2020_train'

# ======================= dataset =======================
if socket.gethostname() == 'z390f':

    # -------------------- train ---------------------
    data_coco_train = '/data3/Datasets/COCO'
    json_coco_train = osjoin(project_dir, 'datasets/json/coco_train.json')

    data_got10k_train = '/data3/Datasets/GOT-10k'
    json_got10k_train = osjoin(project_dir, 'datasets/json/got10k_train.json')

    data_got10k_train_vot = '/data3/Datasets/GOT-10k'
    json_got10k_train_vot = osjoin(project_dir, 'datasets/json/got10k_train_vot.json')

    data_lasot_train = '/data3/Datasets/LaSOT/images'
    json_lasot_train = osjoin(project_dir, 'datasets/json/lasot_train.json')

    data_trackingnet_train = '/data3/Datasets/TrackingNet'
    json_trackingnet_train = osjoin(project_dir, 'datasets/json/trackingnet_train.json')

    # -------------------- lmdb ---------------------
    lmdb_dir = '/data3/LMDB'

    lmdb_coco_train = osjoin(lmdb_dir, 'coco_train')
    lmdb_coco_mask_train = osjoin(lmdb_dir, 'coco_mask_train')
    json_lmdb_coco_train = osjoin(lmdb_coco_train, 'coco_train.json')

    lmdb_coco_val = osjoin(lmdb_dir, 'coco_val')
    lmdb_coco_mask_val = osjoin(lmdb_dir, 'coco_mask_val')
    json_lmdb_coco_val = osjoin(lmdb_coco_val, 'coco_val.json')

    lmdb_got10k_train = osjoin(lmdb_dir, 'got10k_train')
    lmdb_got10k_val = osjoin(lmdb_dir, 'got10k_val')
    json_lmdb_got10k_train = osjoin(lmdb_got10k_train, 'got10k_train.json')
    json_lmdb_got10k_train_vot = osjoin('/data3/LMDB/got10k_train_vot', 'got10k_train_vot.json')
    json_lmdb_got10k_val = osjoin('/data3/LMDB/got10k_val', 'got10k_val.json')

    lmdb_lasot_train = osjoin(lmdb_dir, 'lasot_train')
    lmdb_lasot_val = osjoin(lmdb_dir, 'lasot_val')
    json_lmdb_lasot_train = osjoin(lmdb_lasot_train, 'lasot_train.json')
    json_lmdb_lasot_val = osjoin(lmdb_lasot_val, 'lasot_val.json')

    lmdb_trackingnet_train_p0 = osjoin(lmdb_dir, 'trackingnet_train_p0')
    json_lmdb_trackingnet_train_p0 = osjoin(lmdb_trackingnet_train_p0, 'trackingnet_train_p0.json')

    lmdb_trackingnet_train_p1 = osjoin(lmdb_dir, 'trackingnet_train_p1')
    json_lmdb_trackingnet_train_p1 = osjoin(lmdb_trackingnet_train_p1, 'trackingnet_train_p1.json')

    lmdb_trackingnet_train_p2 = osjoin(lmdb_dir, 'trackingnet_train_p2')
    json_lmdb_trackingnet_train_p2 = osjoin(lmdb_trackingnet_train_p2, 'trackingnet_train_p2.json')

    lmdb_yolo_got10k_train = osjoin(lmdb_dir, 'yolov4_got10k_train')
    lmdb_yolo_got10k_val = osjoin(lmdb_dir, 'yolov4_got10k_val')
    lmdb_yolo_lasot_train = osjoin(lmdb_dir, 'yolov4_lasot_train')
    lmdb_yolo_trackingnet_train_p0 = osjoin(lmdb_dir, 'yolov4_trackingnet_train_p0')
    lmdb_yolo_trackingnet_train_p1 = osjoin(lmdb_dir, 'yolov4_trackingnet_train_p1')

    lmdb_imagenet_train = osjoin(lmdb_dir, 'imagenet_train')
    json_lmdb_imagenet_train = osjoin(lmdb_imagenet_train, 'imagenet_train_label.json')
    lmdb_imagenet_val = osjoin(lmdb_dir, 'imagenet_val')
    json_lmdb_imagenet_val = osjoin(lmdb_imagenet_val, 'imagenet_val_label.json')

    # -------------------- eval ---------------------
    eval_lasot = '/data1/Datasets/LaSOT/dataset/images'
    eval_got10k_val = '/data1/Datasets/GOT-10k/GOT-10k/val'
    eval_got10k_test = '/data1/Datasets/GOT-10k/GOT-10k/test'
    eval_vot20 = '/data1/Dataset/VOT/VOT2020'


else:

    # -------------------- train ---------------------
    data_coco_train = '/home/space/Documents/Datasets/COCO'
    json_coco_train = osjoin(project_dir, 'datasets/json/coco_train.json')

    data_got10k_train = '/home/space/Documents/Datasets/GOT-10k'
    json_got10k_train = osjoin(project_dir, 'datasets/json/got10k_train.json')

    data_got10k_train_vot = '/home/space/Documents/Datasets/GOT-10k'
    json_got10k_train_vot = osjoin(project_dir, 'datasets/json/got10k_train_vot.json')

    data_lasot_train = ''
    json_lasot_train = osjoin(project_dir, 'datasets/json/lasot_train.json')

    data_trackingnet_train = ''
    json_trackingnet_train = osjoin(project_dir, 'datasets/json/trackingnet_train.json')

    # -------------------- lmdb ---------------------
    lmdb_dir = '/media/space/T7/LMDB'

    lmdb_coco_train = osjoin(lmdb_dir, 'coco_train')
    lmdb_coco_mask_train = osjoin(lmdb_dir, 'coco_mask_train')
    json_lmdb_coco_train = osjoin(lmdb_coco_train, 'coco_train.json')

    lmdb_coco_val = osjoin(lmdb_dir, 'coco_val')
    lmdb_coco_mask_val = osjoin(lmdb_dir, 'coco_mask_val')
    json_lmdb_coco_val = osjoin(lmdb_coco_val, 'coco_val.json')

    lmdb_got10k_train = osjoin(lmdb_dir, 'got10k_train')
    lmdb_got10k_val = osjoin(lmdb_dir, 'got10k_val')
    json_lmdb_got10k_train = osjoin(lmdb_got10k_train, 'got10k_train.json')
    json_lmdb_got10k_train_vot = osjoin('/media/space/T7/LMDB/got10k_train_vot', 'got10k_train_vot.json')
    json_lmdb_got10k_val = osjoin('/media/space/T7/LMDB/got10k_val', 'got10k_val.json')

    lmdb_lasot_train = osjoin(lmdb_dir, 'lasot_train')
    lmdb_lasot_val = osjoin(lmdb_dir, 'lasot_val')
    json_lmdb_lasot_train = osjoin(lmdb_lasot_train, 'lasot_train.json')
    json_lmdb_lasot_val = osjoin(lmdb_lasot_val, 'lasot_val.json')

    lmdb_trackingnet_train_p0 = osjoin(lmdb_dir, 'trackingnet_train_p0')
    json_lmdb_trackingnet_train_p0 = osjoin(lmdb_trackingnet_train_p0, 'trackingnet_train_p0.json')

    lmdb_trackingnet_train_p1 = osjoin(lmdb_dir, 'trackingnet_train_p1')
    json_lmdb_trackingnet_train_p1 = osjoin(lmdb_trackingnet_train_p1, 'trackingnet_train_p1.json')

    lmdb_trackingnet_train_p2 = osjoin(lmdb_dir, 'trackingnet_train_p2')
    json_lmdb_trackingnet_train_p2 = osjoin(lmdb_trackingnet_train_p2, 'trackingnet_train_p2.json')

    lmdb_yolo_got10k_train = osjoin(lmdb_dir, 'yolov4_got10k_train')
    lmdb_yolo_got10k_val = osjoin(lmdb_dir, 'yolov4_got10k_val')
    lmdb_yolo_lasot_train = osjoin(lmdb_dir, 'yolov4_lasot_train')
    lmdb_yolo_trackingnet_train_p0 = osjoin(lmdb_dir, 'yolov4_trackingnet_train_p0')
    lmdb_yolo_trackingnet_train_p1 = osjoin(lmdb_dir, 'yolov4_trackingnet_train_p1')

    lmdb_imagenet_train = osjoin(lmdb_dir, 'imagenet_train')
    json_lmdb_imagenet_train = osjoin(lmdb_imagenet_train, 'imagenet_train_label.json')
    lmdb_imagenet_val = osjoin(lmdb_dir, 'imagenet_val')
    json_lmdb_imagenet_val = osjoin(lmdb_imagenet_val, 'imagenet_val_label.json')

    # -------------------- eval ---------------------
    eval_lasot = '/home/space/Documents/Datasets/LaSOT/images'
    eval_got10k_val = '/home/space/Documents/Datasets/GOT-10k/val'
    eval_got10k_test = '/home/space/Documents/Datasets/GOT-10k/test'
    eval_vot20 = ''
