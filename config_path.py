# coding=utf-8
import os

project_path = '/home/space/Documents/wokspace/CVPR2020/code_for_review/'

siam_cfg = os.path.join(project_path, 'modules/pysot/experiments/siamrpn_mobilev2_l234_dwxcorr/config.yaml')
siam_snap = os.path.join(project_path, 'modules/pysot/experiments/siamrpn_mobilev2_l234_dwxcorr/model.pth')

res_model = os.path.join(project_path, 'model/resnet18.pth')

guid_model = os.path.join(project_path, 'model/best_mse.pth')

embed_model = os.path.join(project_path, 'model/best_em.pth')

skim_z = os.path.join(project_path, 'modules/skim/branch_z_3.h5')
skim_s = os.path.join(project_path, 'modules/skim/branch_search_3.h5')

# dataset path
vot18lt = '/data1/Dataset/VOT/LTB35/'
oxuva = '/data1/Dataset/OxUvA/dataset/'
lasot = '/data1/Dataset/LaSOT/dataset/images/'
