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
import mayavi.mlab as mlab


def draw_lidar(pc, color=None, fig=None, bgcolor=(0, 0, 0), pts_scale=1, pts_mode='point', pts_color=None):
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:, 2]
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=pts_color, mode=pts_mode, colormap='gnuplot',
                  scale_factor=pts_scale, figure=fig)

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)

    # draw fov (todo: update to real sensor spec.)
    fov = np.array([  # 45 degree
        [20., 20., 0., 0.],
        [20., -20., 0., 0.],
    ], dtype=np.float64)

    mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig)
    mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig)

    # draw square region
    TOP_Y_MIN = -20
    TOP_Y_MAX = 20
    TOP_X_MIN = 0
    TOP_X_MAX = 40
    TOP_Z_MIN = -2.0
    TOP_Z_MAX = 0.4

    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)

    # mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    # mlab.show()
    return fig

def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

# config_path = "/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/configs/car.fhd.config"
config_path = "/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/configs/nuscenes/all.fhd_Aws.config"
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
ckpt_path = "/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/mayank_all_fhpd/voxelnet-29325.tckpt"
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


info_path = input_cfg.dataset.kitti_info_path
root_path = Path(input_cfg.dataset.kitti_root_path)
with open(info_path, 'rb') as f:
    infos = pickle.load(f)

# points = np.fromfile(
#         '/home/mayank_sati/Documents/point_clouds/nucene_pickle/nuscene/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin',
#         dtype=np.float32)
t = time.time()
# v_path ="/home/mayank_sati/Documents/point_clouds/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin"
# v_path='/home/mayank_sati/Documents/point_clouds/nuscene_v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151622448916.pcd.bin'
v_path='/home/mayank_sati/Documents/point_clouds/000000.bin'
# v_path ="/home/mayank_sati/Documents/point_clouds/nuscene_v1.0-mini/sweeps/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603898164.pcd.bin"
#points = np.fromfile(v_path, dtype=np.float32, count=-1).reshape(-1, 5])
points = np.fromfile(v_path, dtype=np.float32)
# points = points.reshape((-1, 5))[:, :4]
points = points.reshape((-1, 4))
# points = points.reshape((-1, 5))[:, :4]
# voxels, coords, num_points,voxel_num = voxel_generator.generate(points, max_voxels=20000)
#########################################################333
res = voxel_generator.generate(points, max_voxels=80000)
voxels = res["voxels"]
coords = res["coordinates"]
num_points = res["num_points_per_voxel"]
num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
print("voxel_generator_time",(time.time() - t)*1000)
###############################################################
##################################3333
##################################################################
voxel_points=np.squeeze(voxels, axis=1)
fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
pcd_data=voxel_points
draw_lidar(pcd_data, fig=fig)
#####33
counter=0
for  t in coords:
    counter+=1
    mlab.points3d(t[0],t[1],t[2], scale_factor=1.0)
    if counter>100:
        break
#########3
mlab.show()
####################################################################
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
boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
scores_lidar = pred["scores"].detach().cpu().numpy()
##############################3333

threshold=.5
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
# bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
# bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)
# print(bev_map)
# cv2.imshow('color image',bev_map)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##################################3333
##################################################################
fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
pcd_data=points
draw_lidar(pcd_data, fig=fig)
mlab.show()
####################################################################
# dt_annos=2
iLen = scores_lidar.shape[0]
for iIndex in range(iLen):
    loc=boxes_lidar[iIndex][:3]
    dim=boxes_lidar[iIndex][3:6]
    ry=boxes_lidar[iIndex][-1]

    # dim = dt_annos[0]['dim'][iIndex, :]
    # loc = dt_annos[0]['loc'][iIndex, :]
    # score = pred['scores'][iIndex]

    # if (scores_lidar[iIndex] < 0.5):
    #     continue

    # ry = dt_annos[0]['angle'][iIndex]
    w = dim[0]  # box height
    l = dim[1]  # box width
    h = dim[2]  # box length (in meters)
    t = (loc[0], loc[1], loc[2])
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))
    R = rotz(ry)
    corners = np.dot(R, corners)
    x = t[0]
    y = t[1]
    z = t[2]
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z
    corners_3d = corners.T
    color = (0, 1, 0)
    line_width = 1
    mlab.points3d(x, y, z, color=(1, 1, 1), mode='sphere', scale_factor=0.4)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        mlab.plot3d([corners_3d[i, 0], corners_3d[j, 0]], [corners_3d[i, 1], corners_3d[j, 1]],
                    [corners_3d[i, 2], corners_3d[j, 2]], color=color, tube_radius=None, line_width=line_width,
                    figure=fig)
        i, j = k + 4, (k + 1) % 4 + 4
        mlab.plot3d([corners_3d[i, 0], corners_3d[j, 0]], [corners_3d[i, 1], corners_3d[j, 1]],
                    [corners_3d[i, 2], corners_3d[j, 2]], color=color, tube_radius=None, line_width=line_width,
                    figure=fig)
        i, j = k, k + 4
        mlab.plot3d([corners_3d[i, 0], corners_3d[j, 0]], [corners_3d[i, 1], corners_3d[j, 1]],
                    [corners_3d[i, 2], corners_3d[j, 2]], color=color, tube_radius=None, line_width=line_width,
                    figure=fig)

mlab.show()
################################################################3
# ploting second
