import copy
from pathlib import Path
import pickle

import fire

import second.data.kitti_dataset as kitti_ds
import second.data.nuscenes_dataset as nu_ds
from second.data.all_dataset import create_groundtruth_database

def kitti_data_prep(root_path):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database("KittiDataset", root_path, Path(root_path) / "kitti_infos_train.pkl")

def nuscenes_data_prep(root_path, version, dataset_name, max_sweeps=10):
    # nu_ds.create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps)
    name = "infos_train.pkl"
    if version == "v1.0-test":
        name = "infos_test.pkl"
    create_groundtruth_database(dataset_name, root_path, Path(root_path) / name)

if __name__ == '__main__':
    # fire.Fire()
    # if __name__ == '__main__':
    # fire.Fire()
    # save_path_1 = 'save_pkl'
    # save_path_1 = 'save_pkl_mayank_working_for_apollo'
    # save_path_1 = None
    # dataset_path_1 = 'KITTI_DATASET_ROOT'
    # dataset_path_1 = './data/Nuscenes'
    dataset_path_1 = './KITTI_DATASET_ROOT'
    # create_nuscenes_info_file(data_path=dataset_path_1, save_path=save_path_1)
    # create_reduced_point_cloud(data_path=dataset_path_1,save_path=save_path_1)
    # create_nuscenes_groundtruth_database(data_path=dataset_path_1)
    nuscene_path="/home/mayank_sati/Documents/point_clouds/nuscene_v_mayank"
    # nuscene_path="/home/mayank_sati/Documents/point_clouds/nuscene/v1.0-trainval01_blobs"
    # kitti_data_prep(root_path=dataset_path_1)
    # version="v1.0-trainval"
    version="v1.0-mini"
    dataset_name="NuScenesDataset"
    # dataset_name="NuscenesDataset"

    nuscenes_data_prep(nuscene_path,version=version,dataset_name=dataset_name)
# python create_data.py nuscenes_data_prep --data_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --max_sweeps=10
