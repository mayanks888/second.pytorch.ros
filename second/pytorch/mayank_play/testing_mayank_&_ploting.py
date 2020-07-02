import os
import pathlib
import pickle
import shutil
import time
from functools import partial
import glob
import fire
import numpy as np
import torch
from google.protobuf import text_format
# from tensorboardX import SummaryWriter

import torchplus
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                      lr_scheduler_builder, optimizer_builder,
                                      second_builder)
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import ProgressBar
from second.core import box_np_ops
import mayavi.mlab as mlab

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2"
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.uint8, device=device)
        else:
            example_torch[k] = v
    return example_torch


def _predict_kitti_to_file(net,
                           example,
                           result_save_path,
                           class_names,
                           center_limit_range=None,
                           lidar_input=False):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example)
    # t = time.time()
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].data.cpu().numpy()
            box_preds = preds_dict["box3d_camera"].data.cpu().numpy()
            scores = preds_dict["scores"].data.cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].data.cpu().numpy()
            # write pred to file
            box_preds = box_preds[:, [0, 1, 2, 4, 5, 3,
                                      6]]  # lhw->hwl(label file format)
            label_preds = preds_dict["label_preds"].data.cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            result_lines = []
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {
                    'name': class_names[int(label)],
                    'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    'bbox': bbox,
                    'location': box[:3],
                    'dimensions': box[3:6],
                    'rotation_y': box[6],
                    'score': score,
                }
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            1
        #     result_lines = []
        # # result_file = f"{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt"
        # result_str = '\n'.join(result_lines)
        # with open(result_file, 'w') as f:
        #     f.write(result_str)


def predict_kitti_to_anno(net,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None):
    predictions_dicts = net(example)
    return (predictions_dicts)

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

def evaluate(config_path,
             model_dir,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True):
    # model_dir = '../old_trained'
    model_dir = '../trained_model'
    # model_dir = '../nf'
    model_dir=pathlib.Path(model_dir)
    #
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
        # this is to read the config file
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)
##########################################################3
    # loading model
    net = second_builder.build(model_cfg, voxel_generator, target_assigner)
    net.cuda()
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)

    #################################################################333
    # create  anchors
    grid_size = voxel_generator.grid_size
    out_size_factor = model_cfg.rpn.layer_strides[0] // model_cfg.rpn.upsample_strides[0]
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    ret = target_assigner.generate_anchors(feature_map_size)
    anchors = ret["anchors"]
    anchors = anchors.reshape([-1, 7])
    # anchors = np.expand_dims(anchors, axis=0)
    matched_thresholds = ret["matched_thresholds"]
    unmatched_thresholds = ret["unmatched_thresholds"]
    anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
        anchors[:, [0, 1, 3, 4, 6]])
    anchors = np.expand_dims(anchors, axis=0)
    anchor_cache = {
        "anchors": anchors,
        "anchors_bv": anchors_bv,
        "matched_thresholds": matched_thresholds,
        "unmatched_thresholds": unmatched_thresholds,
    }
    ##############################################################################
    # Loading bin file
    max_voxels=20000
    # scan = np.fromfile('../data/Org/training/velodyne_reduced/000018.bin',dtype=np.float32)
    # scan = np.fromfile('../data/Nuscenes/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin', dtype=np.float32)
    scan = np.fromfile('.000009.bin', dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :4]
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size
    # [352, 400]

    voxels, coordinates, num_points = voxel_generator.generate(
        points, max_voxels)
    anchor_area_threshold=1
    if anchor_area_threshold >= 0:
        coors = coordinates
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
            coors, tuple(grid_size[::-1][1:]))
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = box_np_ops.fused_get_anchors_area(
            dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
        anchors_mask = anchors_area > anchor_area_threshold
    kl = np.zeros((coordinates.shape[0], 1))
    coordinates = np.append(kl, coordinates, axis=1)
    example = {

        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": np.array([voxels.shape[0]], dtype=np.int64),
        "anchors": anchors,
        "anchors_mask": anchors_mask

    }
    example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / "step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    dt_annos = []
    global_set = None
    print("Generate output labels...")
    example = example_convert_to_torch(example, float_dtype)
    if pickle_result:
            dt_annos += predict_kitti_to_anno(net, example, class_names, center_limit_range, model_cfg.lidar_input, global_set)
    else:
            _predict_kitti_to_file(net, example, result_path_step, class_names,
                                   center_limit_range, model_cfg.lidar_input)
    if not predict_test:
        if not pickle_result:
            dt_annos = kitti.get_label_annos(result_path_step)
        if pickle_result:
            with open(result_path_step / "result_pp.pkl", 'wb') as f:
                pickle.dump(dt_annos, f)

def evaluate2(config_path,
             model_dir,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True):
    # model_dir = '../trained_model'
    model_dir = '../old_trained_FIRST'
    # model_dir = '../mayank_pc_trained'
    model_dir=pathlib.Path(model_dir)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
        # this is to read the config file
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)
##########################################################3
    # loading model
    net = second_builder.build(model_cfg, voxel_generator, target_assigner)
    net.cuda()
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)

    #################################################################333
    # create  anchors
    grid_size = voxel_generator.grid_size
    out_size_factor = model_cfg.rpn.layer_strides[0] // model_cfg.rpn.upsample_strides[0]
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    ret = target_assigner.generate_anchors(feature_map_size)
    anchors = ret["anchors"]
    anchors = anchors.reshape([-1, 7])
    # anchors = np.expand_dims(anchors, axis=0)
    matched_thresholds = ret["matched_thresholds"]
    unmatched_thresholds = ret["unmatched_thresholds"]
    anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
        anchors[:, [0, 1, 3, 4, 6]])
    anchors = np.expand_dims(anchors, axis=0)
    anchor_cache = {
        "anchors": anchors,
        "anchors_bv": anchors_bv,
        "matched_thresholds": matched_thresholds,
        "unmatched_thresholds": unmatched_thresholds,
    }
    ##############################################################################
    # Loading bin file
    max_voxels=20000
    # scan = np.fromfile('../data/Org/training/velodyne_reduced/000018.bin',dtype=np.float32)
    # scan = np.fromfile('/home/mayank_sati/pycharm_projects/second_cd/second/data/velodyne/000016.bin', dtype=np.float32)
    # scan = np.fromfile('/home/mayank_sati/Downloads/002_00000000.bin', dtype=np.float32)
    # scan = np.fromfile('/home/mayank_sati/pycharm_projects/second_cd/second/data/velodyne_reduced/000012.bin', dtype=np.float32)
    # scan = np.fromfile('/home/mayank_sati/pycharm_projects/second_cd/second/data/Nuscenes2/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915255448617.pcd.bin', dtype=np.float32)
    # scan = np.fromfile('/home/mayank_sati/pycharm_projects/second_cd/second/data/Nuscenes2/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915265947124.pcd.bin', dtype=np.float32)
    # scan = np.fromfile('', dtype=np.float32)
    #
    # scan = np.fromfile('/home/mayank_sati/Documents/kittivelodyne/pc/0000000111.bin', dtype=np.float32)
    scan = np.fromfile(
        '/home/mayank_sati/Documents/point_clouds/nucene_pickle/nuscene/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin',
        dtype=np.float32)
    points = scan.reshape((-1, 5))[:, :4]
    # points = scan.reshape((-1, 4))[:, :4]
    # points = scan.reshape((-1, 4))
    voxel_size = voxel_generator.voxel_size

    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size
    # [352, 400]
    ############################################################################3

    #############################################################################
    voxels, coordinates, num_points = voxel_generator.generate(points, max_voxels)
    anchor_area_threshold=1
    if anchor_area_threshold >= 0:
        coors = coordinates
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(coors, tuple(grid_size[::-1][1:]))
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = box_np_ops.fused_get_anchors_area(dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
        anchors_mask = anchors_area > anchor_area_threshold

    kl = np.zeros((coordinates.shape[0], 1))
    coordinates = np.append(kl, coordinates, axis=1)
    example = {

        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": np.array([voxels.shape[0]], dtype=np.int64),
        "anchors": anchors,
        "anchors_mask": anchors_mask

    }
    example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / "step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    dt_annos = []
    global_set = None
    print("Generate output labels...")
    example = example_convert_to_torch(example, float_dtype)
    dt_annos += predict_kitti_to_anno(net, example, class_names, center_limit_range, model_cfg.lidar_input, global_set)
    with open(result_path_step / "result_pp1.pkl", 'wb') as f:
                pickle.dump(dt_annos, f)
    ##################################################################
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
    pcd_data=points
    draw_lidar(pcd_data, fig=fig)
    # mlab.show()
    ####################################################################
    iLen = dt_annos[0]['scores'].shape[0]
    for iIndex in range(iLen):
        dim = dt_annos[0]['dim'][iIndex, :]
        loc = dt_annos[0]['loc'][iIndex, :]
        score = dt_annos[0]['scores'][iIndex]

        if (score < 0.3):
            continue

        ry = dt_annos[0]['angle'][iIndex]
        w = dim[0]  # box height
        h = dim[1]  # box width
        l = dim[2]  # box length (in meters)
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
    ################################################################3
    # ploting second



    points2 = points
    points2[:, 0] = -points2[:, 0]
    voxels2, coordinates2, num_points2 = voxel_generator.generate(points2, max_voxels)
    anchor_area_threshold = 1
    if anchor_area_threshold >= 0:
        coors2 = coordinates2
        dense_voxel_map2 = box_np_ops.sparse_sum_for_anchors_mask(coors2, tuple(grid_size[::-1][1:]))
        dense_voxel_map2 = dense_voxel_map2.cumsum(0)
        dense_voxel_map2 = dense_voxel_map2.cumsum(1)
        anchors_area2 = box_np_ops.fused_get_anchors_area(dense_voxel_map2, anchors_bv, voxel_size, pc_range, grid_size)
        anchors_mask2 = anchors_area2 > anchor_area_threshold
    kl2 = np.zeros((coordinates2.shape[0], 1))
    coordinates2 = np.append(kl2, coordinates2, axis=1)
    example = {

        'voxels': voxels2,
        'num_points': num_points2,
        'coordinates': coordinates2,
        "num_voxels": np.array([voxels2.shape[0]], dtype=np.int64),
        "anchors": anchors,
        "anchors_mask": anchors_mask2

    }
    example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
    net.eval()
    dt_annos = []
    global_set = None
    print("Generate output labels2...")
    example = example_convert_to_torch(example, float_dtype)
    dt_annos += predict_kitti_to_anno(net, example, class_names, center_limit_range, model_cfg.lidar_input, global_set)
    with open(result_path_step / "result_pp2.pkl", 'wb') as f:
        pickle.dump(dt_annos, f)
    ####################################################33
    # plot negative axis
    iLen = dt_annos[0]['scores'].shape[0]
    for iIndex in range(iLen):
        dim =dt_annos[0]['dim'][iIndex, :]
        loc = dt_annos[0]['loc'][iIndex, :]
        score = dt_annos[0]['scores'][iIndex]

        # dim = boxes3d[0]['dim'][iIndex, :]
        # loc = boxes3d[0]['loc'][iIndex, :]
        # score = boxes3d[0]['scores'][iIndex]
        if (score < 0.2):
            continue
        ry = dt_annos[0]['angle'][iIndex]  # +np.pi
        w = dim[0]  # box height
        h = dim[1]  # box width
        l = dim[2]  # box length (in meters)
        t = (loc[0], loc[1], loc[2])
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        R = rotz(ry)
        corners = np.dot(R, corners)
        x = -t[0]
        y = t[1]
        z = t[2]
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z
        corners_3d = corners.T
        color = (0, 1, 0)
        line_width = 1
        mlab.points3d(x, y, z, color=(1, 1, 1), mode='sphere', scale_factor=0.4)
        # mlab.text3d(-x, y, z, '%d' % iIndex, scale=(1, 1, 1), color=color, figure=fig)
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

    ######################################################################

def evaluate3(config_path,
             model_dir,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True):
    model_dir = '../old_trained_FIRST'
    model_dir=pathlib.Path(model_dir)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
        # this is to read the config file
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    class_names = list(input_cfg.class_names)
    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)
##########################################################3
    # loading model
    net = second_builder.build(model_cfg, voxel_generator, target_assigner)
    net.cuda()
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)

    #################################################################333
    # create  anchors
    grid_size = voxel_generator.grid_size
    out_size_factor = model_cfg.rpn.layer_strides[0] // model_cfg.rpn.upsample_strides[0]
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    ret = target_assigner.generate_anchors(feature_map_size)
    anchors = ret["anchors"]
    anchors = anchors.reshape([-1, 7])
    # anchors = np.expand_dims(anchors, axis=0)
    matched_thresholds = ret["matched_thresholds"]
    unmatched_thresholds = ret["unmatched_thresholds"]
    anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
        anchors[:, [0, 1, 3, 4, 6]])
    anchors = np.expand_dims(anchors, axis=0)
    anchor_cache = {
        "anchors": anchors,
        "anchors_bv": anchors_bv,
        "matched_thresholds": matched_thresholds,
        "unmatched_thresholds": unmatched_thresholds,
    }
    max_voxels=20000
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size
    anchor_area_threshold = 1
    # [352, 400]
    ##############################################################################
    # Loading bin file
    pcd_names = glob.glob('../data/Nuscenes2/*.pcd.bin')
    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)
    if train_cfg.enable_mixed_precision:
            float_dtype = torch.float16
    else:
            float_dtype = torch.float32

    for idx, pcd_name in enumerate(pcd_names):
        scan = np.fromfile(pcd_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, :4]
        voxels, coordinates, num_points = voxel_generator.generate(points, max_voxels)
        if anchor_area_threshold >= 0:
            coors = coordinates
            dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(coors, tuple(grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = box_np_ops.fused_get_anchors_area(dense_voxel_map, anchors_bv, voxel_size, pc_range,
                                                             grid_size)
            anchors_mask = anchors_area > anchor_area_threshold

        kl = np.zeros((coordinates.shape[0], 1))
        coordinates = np.append(kl, coordinates, axis=1)
        example = {

            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coordinates,
            "num_voxels": np.array([voxels.shape[0]], dtype=np.int64),
            "anchors": anchors,
            "anchors_mask": anchors_mask

        }
        example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        net.eval()
        result_path_step = "../results/"
        # result_path_step.mkdir(parents=True, exist_ok=True)
        t = time.time()
        dt_annos = []
        global_set = None
        print("Generate output labels...")
        example = example_convert_to_torch(example, float_dtype)
        dt_annos += predict_kitti_to_anno(net, example, class_names, center_limit_range, model_cfg.lidar_input,global_set)
        pcStrFile1 = pcd_name.split('/')
        pcStrFile1 = pcStrFile1[len(pcStrFile1) - 1] + "_1.pkl"
        pcStrFile1 = result_path_step+pcStrFile1
        with open(pcStrFile1, 'wb') as f:
            pickle.dump(dt_annos, f)
        points2 = points
        points2[:, 0] = -points2[:, 0]
        voxels2, coordinates2, num_points2 = voxel_generator.generate(points2, max_voxels)
        anchor_area_threshold = 1
        if anchor_area_threshold >= 0:
            coors2 = coordinates2
            dense_voxel_map2 = box_np_ops.sparse_sum_for_anchors_mask(coors2, tuple(grid_size[::-1][1:]))
            dense_voxel_map2 = dense_voxel_map2.cumsum(0)
            dense_voxel_map2 = dense_voxel_map2.cumsum(1)
            anchors_area2 = box_np_ops.fused_get_anchors_area(dense_voxel_map2, anchors_bv, voxel_size, pc_range,grid_size)
            anchors_mask2 = anchors_area2 > anchor_area_threshold
        kl2 = np.zeros((coordinates2.shape[0], 1))
        coordinates2 = np.append(kl2, coordinates2, axis=1)
        example = {

            'voxels': voxels2,
            'num_points': num_points2,
            'coordinates': coordinates2,
            "num_voxels": np.array([voxels2.shape[0]], dtype=np.int64),
            "anchors": anchors,
            "anchors_mask": anchors_mask2
        }
        example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        net.eval()
        dt_annos = []
        global_set = None
        print("Generate output labels2...")
        example = example_convert_to_torch(example, float_dtype)
        dt_annos += predict_kitti_to_anno(net, example, class_names, center_limit_range, model_cfg.lidar_input,global_set)
        pcStrFile2 = pcd_name.split('/')
        pcStrFile2 = pcStrFile2[len(pcStrFile2) - 1] + "_2.pkl"
        pcStrFile2 = result_path_step + pcStrFile2
        with open(pcStrFile2, 'wb') as f:
            pickle.dump(dt_annos, f)

if __name__ == '__main__':
    # fire.Fire(=)

    # config_path = '../configs/pointpillars/car/xyres_16_nuscenes.proto'
    config_path = '../configs/pointpillars/car/xyres_16_nuscenes_2.proto'
    # model_dir='../trained_dodel_dir'
    # model_dir = '../trained_model'
    model_dir = '../mayank_pc_trained'
    # result_path=
    # train(config_path, model_dir)
    evaluate2(config_path, model_dir)
    # evaluate3(config_path, model_dir)

