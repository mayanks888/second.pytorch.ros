import pickle
import numpy as np
import mayavi.mlab as mlab


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
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    # mlab.show()
    return fig

# datapath_file ='/home/mayank_sati/pycharm_projects/pytorch/second_nuscene_mayank/second/save_pkl/nuscenes_infos_train.pkl'
# datapath_file ='/home/mayank_sati/pycharm_projects/tensorflow/traffic_light_detection_classification-master/traffic_light_classification/autokeras/model_file/test_autokeras_model.pkl'
# datapath_file ='/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/point_pp_nuscene/eval_results/step_140670/result.pkl'
datapath_file ='/home/mayank_sati/Documents/point_clouds/nuscene_v_mayank/infos_train.pkl'
# datapath_file ='/home/mayank_sati/Documents/point_clouds/nuscene_v1.0-mini/infos_val.pkl'
# datapath_file ='/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/pytorch/38_lidar.pkl'
boxes = pickle.load(open(datapath_file, "rb"))
print(1)

# for info in boxes['infos'][2]['sweeps']:
# for info in boxes['infos']:
for iIndex, info in enumerate(boxes['infos'], start=0):
    if iIndex==0:
        continue
    print(info)
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
    fig = draw_lidar_simple(points)
    mlab.show()