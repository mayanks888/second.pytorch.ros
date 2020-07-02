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
import os

def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def cart2hom(pts_3d):
    ''' Input: nx3 points in Cartesian
        Oupput: nx4 points in Homogeneous by pending 1
    '''
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom


def project_velo_to_ref(pts_3d_velo):
    pts_3d_velo = cart2hom(pts_3d_velo)  # nx4
    return np.dot(pts_3d_velo, np.transpose(V2C))


def project_ref_to_velo(pts_3d_ref):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(C2V))


def project_rect_to_ref(pts_3d_rect):
    ''' Input and Output are nx3 points '''
    # t1=np.linalg.inv(R0)
    # t2=pts_3d_rect
    # t3=np.dot(t1,t2)
    # t4 = np.transpose(t3)
    return np.transpose(np.dot(np.linalg.inv(R0), np.transpose(pts_3d_rect)))


def project_ref_to_rect(pts_3d_ref):
    ''' Input and Output are nx3 points '''
    return np.transpose(np.dot(R0, np.transpose(pts_3d_ref)))


def project_rect_to_velo(pts_3d_rect):
    ''' Input: nx3 points in rect camera coord.
        Output: nx3 points in velodyne coord.
    '''
    pts_3d_ref = project_rect_to_ref(pts_3d_rect)
    return project_ref_to_velo(pts_3d_ref)


def project_velo_to_rect(pts_3d_velo):
    pts_3d_ref = h.project_velo_to_ref(pts_3d_velo)
    return project_ref_to_rect(pts_3d_ref)

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


def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr

def pointCloudToBirdsEyeView(ax2, velo, bb3d):
    ax2.set_xlim (-10,10)
    ax2.set_ylim (-5,35)
    hmax = velo[:,2].max()
    hmin = velo[:,2].min()
    hmean = velo[:, 2].mean()
    hmeadian = np.median ( velo[:, 2] )
    hstd = np.std(velo[:, 2])
    #print ('scalledh', hmax, hmean, hmeadian, hmin, hstd, scalledh.shape, scalledh[:10])
    norm = colors.Normalize(hmean-2*hstd, hmean+2*hstd, clip=True)
    sc2= ax2.scatter(-velo[:,1],
             velo[:,0],
             s = 1,
             c=velo[:,2],
             cmap = 'viridis',
             norm=norm,
             marker = ".",
             )
    ax2.scatter(-bb3d[:,1],
             bb3d[:,0],
             c='red')
    ax2.set_facecolor('xkcd:grey')
    plt.colorbar(sc2)
    plt.waitforbuttonpress()

def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    #draw points
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=None, mode='point', colormap = 'gnuplot', scale_factor=1, figure=fig)
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=2)
    #draw axis
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    # mlab.show()
    return fig

def draw_gt_boxes3d(gt_boxes3d,iIndex,fig,color=(1,1,1), line_width=1, draw_text=True, text_scale=(1,1,1), color_list=None):
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%iIndex, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    # mlab.show()
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig

def hv_in_range(x, y, z, fov, fov_type='h'):
    """
    Extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit
    Args:
    `x`:velodyne points x array
    `y`:velodyne points y array
    `z`:velodyne points z array
    `fov`:a two element list, e.g.[-45,45]
    `fov_type`:the fov type, could be `h` or 'v',defualt in `h`
    Return:
    `cond`:condition of points within fov or not
    Raise:
    `NameError`:"fov type must be set between 'h' and 'v' "
    """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if fov_type == 'h':
        return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi/180), np.arctan2(y, x) < (-fov[0] * np.pi/180))
    elif fov_type == 'v':
        return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180), np.arctan2(z, d) > (fov[0] * np.pi / 180))
    else:
        raise NameError("fov type must be set between 'h' and 'v' ")
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

class_names = target_assigner.classes

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

##############################333
# datapath_file ='/home/mayank_sati/Documents/point_clouds/nuscene_v_mayank/infos_train.pkl'
datapath_file ='/home/mayank_sati/Documents/point_clouds/nuscene_v_mayank/infos_val.pkl'
# datapath_file ='/home/mayank_sati/Documents/point_clouds/nuscene_v1.0-mini/infos_val.pkl'
# datapath_file ='/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/pytorch/38_lidar.pkl'
boxes = pickle.load(open(datapath_file, "rb"))
print(1)

# for info in boxes['infos'][2]['sweeps']:
# for info in boxes['infos']:
for iIndex, info in enumerate(boxes['infos'], start=0):
    if iIndex==0:
        continue
    # print(info)
    lidar_path=info['lidar_path']
#this is how they did it superimposing
    # lidar_path = lidar_path
    points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
    points[:, 3] /= 255
    points[:, 4] = 0
    sweep_points_list = [points]
    ts = info["timestamp"] / 1e6

    for sweep in info["sweeps"]:
            points_sweep = np.fromfile(str(sweep["lidar_path"]), dtype=np.float32, count=-1).reshape([-1, 5])
            sweep_ts = sweep["timestamp"] / 1e6
            points_sweep[:, 3] /= 255
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep["sweep2lidar_rotation"].T
            points_sweep[:, :3] += sweep["sweep2lidar_translation"]
            points_sweep[:, 4] = ts - sweep_ts
            sweep_points_list.append(points_sweep)

            points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]
    # fig = draw_lidar_simple(points)
    # mlab.show()
##################################3

            res = voxel_generator.generate(points, max_voxels=50000)
            voxels = res["voxels"]
            coords = res["coordinates"]
            num_points = res["num_points_per_voxel"]
            num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
            # print("voxel_generator_time",(time.time() - t)*1000)
            ###############################################################
            # print(voxels.shape)
            # add batch idx to coords
            coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
            voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
            coords = torch.tensor(coords, dtype=torch.int32, device=device)
            num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
            # print("conversion time",(time.time() - t)*1000)
            example = {"anchors": anchors, "voxels": voxels, "num_points": num_points, "coordinates": coords,}
            t2 = time.time()
            pred = net(example)[0]
            # print("prediction",(time.time() - t2)*1000)
            # print("total_time",(time.time() - t)*1000)
            boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
            scores_lidar = pred["scores"].detach().cpu().numpy()
            labels_lidar = pred["label_preds"].detach().cpu().numpy()
            ##############################3333
            threshold=.3
            keep = np.where((scores_lidar >= threshold))[0]
            scores_lidar = scores_lidar[keep]
            boxes_lidar = boxes_lidar[keep]
            labels_lidar = labels_lidar[keep]
            #######################################
            fig = draw_lidar_simple(points)
            # for box3d in boxes_lidar:
            for iIndex, box3d in enumerate(boxes_lidar, start=0):
                location = box3d[:3]
                dimensn = box3d[3:6]
                angle = box3d[-1]

                h = dimensn[2]  # box height
                w = dimensn[1]  # box width
                l = dimensn[0]  # box length (in meters)
                t = (location[0], location[1], location[2])
                x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
                y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
                z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
                corners = np.vstack((x_corners, y_corners, z_corners))
                ry = angle
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
                mlab.text3d(x, y, z, class_names[labels_lidar[iIndex]], scale=(.5, .5, .5))
                mlab.text3d(x+1, y+1, z+1, str(round(scores_lidar[iIndex],2)), scale=(.5, .5, .5))

                # mlab.title('car')
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