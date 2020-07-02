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
import natsort
import argparse

def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    #draw points
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=None, mode='point', colormap = 'gnuplot', scale_factor=1, figure=fig)
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.5)
    #draw axis
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    mlab.view(azimuth=90, elevation=60, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=50.0, figure=fig)
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


##################33333
if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Pointpillars inference')
    PARSER.add_argument('-config_pth', '--config_path', default='../configs/nuscenes/all.pp.largea.config')
    PARSER.add_argument('-ckpt', '--ckpt_path', default='../checkpoint/voxelnet-140670.tckpt')
    PARSER.add_argument('-i', '--input_folder', default='../input_point_clouds')
    ARGS = PARSER.parse_args()

    config_path=ARGS.config_path
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    input_cfg = config.eval_input_reader
    model_cfg = config.model.second

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ARGS.ckpt_path
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

    info_path = input_cfg.dataset.kitti_info_path
    root_path = Path(input_cfg.dataset.kitti_root_path)
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    t = time.time()
    input_folder=ARGS.input_folder
    for root, _, filenames in os.walk(input_folder):
        if (len(filenames) == 0):
            print("Input folder is empty")
            # return 1
        filenames = natsort.natsorted(filenames, reverse=False)

        # time_start = time.time()
        for filename in filenames:
            # try:
                filename=input_folder+'/'+filename
                points = np.fromfile(filename, dtype=np.float32)
                points = points.reshape((-1, 5))[:, :4]
                # points = points.reshape((-1, 4))
                # points = points.reshape((-1, 5))[:, :4]
                ####################################################3
                # points = np.fromfile(str(v_path), dtype=np.float32, count=-1).reshape([-1, 5])
                points[:, 3] /= 255
                # points[:, 4] = 0
                #########################################################333
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
                ##############################3333
                threshold=.3
                keep = np.where((scores_lidar >= threshold))[0]
                scores_lidar = scores_lidar[keep]
                boxes_lidar = boxes_lidar[keep]
                #######################################
                fig = draw_lidar_simple(points)
                for box3d in boxes_lidar:
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
                    # mlab.points3d(x, y, z, color=(1, 1, 1), mode='sphere', scale_factor=0.4)
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
                mlab.view(azimuth=270, elevation=30, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=120.0, roll=2,figure=fig)
                mlab.show()
