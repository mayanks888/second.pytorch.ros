import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import torch
from google.protobuf import text_format
from second.utils import simplevis
from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.utils import config_tool
import time
import cv2
# config_path = "/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/configs/car.fhd.config"
# config_path = "/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/configs/nuscenes/all.fhd_Aws.config"
config_path = "/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/configs/nuscenes/all.pp.largea.config"
config = pipeline_pb2.TrainEvalPipelineConfig()
with open(config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, config)
input_cfg = config.eval_input_reader
model_cfg = config.model.second
# config_tool.change_detection_range(model_cfg, [-50, -50, 50, 50])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print("i was here")
# /home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/mayank_pc_trained/voxelnet-5865.tckpt
# ckpt_path = "/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/mayank_all_fhpd/voxelnet-29325.tckpt"
ckpt_path = "/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/point_pp_nuscene/voxelnet-140670.tckpt"
# ckpt_path = "/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/eval_result/pretrained_models_v1.5/pp_model_for_nuscenes_pretrain/voxelnet-296960.tckpt"
net = build_network(model_cfg).to(device).eval()
net.load_state_dict(torch.load(ckpt_path))
target_assigner = net.target_assigner
voxel_generator = net.voxel_generator


grid_size = voxel_generator.grid_size
feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
feature_map_size = [*feature_map_size, 1][::-1]

anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
anchors = anchors.view(1, -1, 7)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
feature_map_size = [1, 50, 50]
ret = target_assigner.generate_anchors(feature_map_size)
class_names = target_assigner.classes
anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
anchors_list = []
for k, v in anchors_dict.items():
    anchors_list.append(v["anchors"])

# anchors = ret["anchors"]
anchors = np.concatenate(anchors_list, axis=0)
anchors = anchors.reshape([-1, target_assigner.box_ndim])
assert np.allclose(anchors, ret["anchors"].reshape(-1, target_assigner.box_ndim))
matched_thresholds = ret["matched_thresholds"]
unmatched_thresholds = ret["unmatched_thresholds"]
# anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
anchors_bv=2
anchor_cache = {
    "anchors": anchors,
    "anchors_bv": anchors_bv,
    "matched_thresholds": matched_thresholds,
    "unmatched_thresholds": unmatched_thresholds,
    "anchors_dict": anchors_dict,
}
anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
anchors = anchors.view(1, -1, 7)
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

info_path = input_cfg.dataset.kitti_info_path
root_path = Path(input_cfg.dataset.kitti_root_path)
with open(info_path, 'rb') as f:
    infos = pickle.load(f)

# points = np.fromfile(
#         '/home/mayank_sati/Documents/point_clouds/nucene_pickle/nuscene/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin',
#         dtype=np.float32)
t = time.time()
# v_path ="/home/mayank_sati/Documents/point_clouds/000000.bin"
v_path='/home/mayank_sati/Documents/point_clouds/nuscene_v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151622448916.pcd.bin'
# v_path='/home/mayank_sati/Documents/point_clouds/nuscene_v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151622448916.pcd.bin'
#points = np.fromfile(v_path, dtype=np.float32, count=-1).reshape(-1, 5])
points = np.fromfile(v_path, dtype=np.float32)
points = points.reshape((-1, 5))[:, :4]
# points = points.reshape((-1, 4))
# points = points.reshape((-1, 5))[:, :4]
# voxels, coords, num_points,voxel_num = voxel_generator.generate(points, max_voxels=20000)

####################################################3
# points = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([-1, 5])
points[:, 3] /= 255
# points[:, 4] = 0
#########################################################333
res = voxel_generator.generate(points, max_voxels=80000)
voxels = res["voxels"]
coords = res["coordinates"]
num_points = res["num_points_per_voxel"]
num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
print("voxel_generator_time",(time.time() - t)*1000)
###############################################################
# print(voxels.shape)
# add batch idx to coords
coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
coords = torch.tensor(coords, dtype=torch.int32, device=device)
num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
print("conversion time",(time.time() - t)*1000)
example = {"anchors": anchors, "voxels": voxels, "num_points": num_points, "coordinates": coords,}
t2 = time.time()
pred = net(example)[0]
print("prediction",(time.time() - t2)*1000)
print("total_time",(time.time() - t)*1000)
boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
scores_lidar = pred["scores"].detach().cpu().numpy()
##############################3333
threshold=.3
keep = np.where((scores_lidar >= threshold))[0]
scores_lidar = scores_lidar[keep]
boxes_lidar = boxes_lidar[keep]
#######################################
with open("result_nu1_scores.pkl", 'wb') as f:
    pickle.dump(scores_lidar, f)
with open("result_nu1.pkl", 'wb') as f:
    pickle.dump(boxes_lidar, f)

vis_voxel_size = [0.1, 0.1, 0.1]
vis_point_range = [-50, -30, -3, 50, 30, 1]
bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)
print(bev_map)
cv2.imshow('color image',bev_map)
cv2.waitKey(0)
cv2.destroyAllWindows()