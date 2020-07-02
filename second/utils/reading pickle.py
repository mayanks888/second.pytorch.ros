import pickle
# datapath_file ='/home/mayank_sati/pycharm_projects/pytorch/second_nuscene_mayank/second/save_pkl/nuscenes_infos_train.pkl'
# datapath_file ='/home/mayank_sati/pycharm_projects/tensorflow/traffic_light_detection_classification-master/traffic_light_classification/autokeras/model_file/test_autokeras_model.pkl'
datapath_file ='/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/point_pp_nuscene/eval_results/step_140670/result.pkl'
# datapath_file ='/home/mayank_sati/Documents/point_clouds/nuscene_v1.0-mini/infos_val.pkl'
boxes = pickle.load(open(datapath_file, "rb"))
print(1)
# import mayavi.mlab as mlab
#
# fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
#         pcd_data = gt_points
#         print(pcd_data.shape)
#         # pcd_data = points
#         draw_lidar(pcd_data, fig=fig)
#         mlab.show()