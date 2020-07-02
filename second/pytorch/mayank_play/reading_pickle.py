import pickle
# datapath_file ='/home/mayank_sati/pycharm_projects/pytorch/second_nuscene_mayank/second/save_pkl/nuscenes_infos_train.pkl'
# datapath_file ='/home/mayank_sati/pycharm_projects/tensorflow/traffic_light_detection_classification-master/traffic_light_classification/autokeras/model_file/test_autokeras_model.pkl'
# datapath_file ='/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/point_pp_nuscene/eval_results/step_140670/result.pkl'
datapath_file ='/home/mayank_sati/Documents/point_clouds/nuscene_v_mayank/kitti_dbinfos_train.pkl'
# datapath_file ='/home/mayank_sati/Documents/point_clouds/nuscene_v1.0-mini/infos_val.pkl'
# datapath_file ='/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/pytorch/38_lidar.pkl'
boxes = pickle.load(open(datapath_file, "rb"))
print(1)

# metadata= boxes["metadata"]
# print(metadata)
# dt_annos1=[]
# # infos=[]
# for dt_annos in boxes['infos']:
#     with open("38_result.pkl", 'wb') as f:
#         print(dt_annos['lidar_path'])
#         if dt_annos['lidar_path']=='/home/mayank_sati/Documents/point_clouds/nuscene_v1.0-mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151622448916.pcd.bin':
#             dt_annos1.append(dt_annos)
#             data = {
#                 "infos": dt_annos1,
#                 'metadata':metadata
#             }
#             with open("38_lidar.pkl", 'wb') as f:
#
#                 pickle.dump(data, f)