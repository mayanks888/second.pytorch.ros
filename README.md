## Pointpillar : 3D Object Detection on Point Cloud
###this project is forked from nutonomy/second.pytorch, and modified for ros environment.
Pointpillars : https://arxiv.org/abs/1812.05784

## Requirenment:



1.  Install dependence python packages
    ######anaconda environment is prefered for the pointpillars model
   
    You can find anaconda env yaml  in `requirements.yml`        
   Install using `conda env update -f requirements.yml`
   
2. Activate anaconda environment :
    `source activate <anaconda env`>
    
    eg: ` souce acctivate pointpillars_env`

3. Follow instructions in [spconv](https://github.com/traveller59/spconv) to install spconv. 
    Note: make sure to install spconv library inside the dedicated conda environment 
   
    eg `pip install spconv-1.1-cp37-cp37m-linux_x86_64.whl -t /home/mayank_sati/anaconda/envs/pointpillars_env_2/lib/python3.7/site-packages --upgrade
`

4. If you want to use NuScenes dataset, you need to install [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit).

5. If you want to train with fp16 mixed precision (train faster in RTX series, Titan V/RTX and Tesla V100, but I only have 1080Ti), you need to install [apex](https://github.com/NVIDIA/apex).



## Prepare dataset


* [NuScenes](https://www.nuscenes.org) Dataset preparation

NuScene Dataset preparation
Download NuScenes dataset from :  https://www.nuscenes.org/download 
Download devkits: https://github.com/nutonomy/nuscenes-devkit
Note: To download nuScenes you need to go to the Download page, create an account, After logging in you will see multiple archives. For the devkit to work you will need to download all archives. Please unpack the archives to the /data/sets/nuscenes folder *without* overwriting folders that occur in multiple archives. Eventually you should have the following folder structure:



Download NuScenes dataset:
```plain
└── NUSCENES_TRAINVAL_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       └── v1.0-trainval <-- metadata and annotations
└── NUSCENES_TEST_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       └── v1.0-test     <-- metadata
```
####go to path : PointPillars/second

Then run
```bash
python create_data.py nuscenes_data_prep --data_path=NUSCENES_TRAINVAL_DATASET_ROOT --version="v1.0-trainval" --max_sweeps=10
python create_data.py nuscenes_data_prep --data_path=NUSCENES_TEST_DATASET_ROOT --version="v1.0-test" --max_sweeps=10
--dataset_name="NuscenesDataset"
```
This will create gt database **without velocity**. to add velocity, use dataset name ```NuscenesDatasetVelo```.

* Modify config file

There is some path need to be configured in config file:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/dataset_dbinfos_train.pkl"
    ...
  }
  dataset: {
    dataset_class_name: "DATASET_NAME"
    kitti_info_path: "/path/to/dataset_infos_train.pkl"
    kitti_root_path: "DATASET_ROOT"
  }
}
...
eval_input_reader: {
  ...
  dataset: {
    dataset_class_name: "DATASET_NAME"
    kitti_info_path: "/path/to/dataset_infos_val.pkl"
    kitti_root_path: "DATASET_ROOT"
  }
}
```

## Usage

### train

I recommend to use script.py to train and eval. see script.py for more details.

#### train with single GPU

```bash
python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir
```

#### train with multiple GPU (need test, I only have one GPU)

Assume you have 4 GPUs and want to train with 3 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,3 python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir --multi_gpu=True
```

Note: The batch_size and num_workers in config file is per-GPU, if you use multi-gpu, they will be multiplied by number of GPUs. Don't modify them manually.

You need to modify total step in config file. For example, 50 epochs = 15500 steps for car.lite.config and single GPU, if you use 4 GPUs, you need to divide ```steps``` and ```steps_per_eval``` by 4.

#### train with fp16 (mixed precision)

Modify config file, set enable_mixed_precision to true.

* Make sure "/path/to/model_dir" doesn't exist if you want to train new model. A new directory will be created if the model_dir doesn't exist, otherwise will read checkpoints in it.

* training process use batchsize=6 as default for 1080Ti, you need to reduce batchsize if your GPU has less memory.

* Currently only support single GPU training, but train a model only needs 20 hours (165 epoch) in a single 1080Ti and only needs 50 epoch to reach 78.3 AP with super converge in car moderate 3D in Kitti validation dateset.

### evaluate

```bash
python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=/path/to/model_dir --measure_time=True --batch_size=1
```

* detection result will saved as a result.pkl file in model_dir/eval_results/step_xxx or save as official KITTI label format if you use --pickle_result=False.

###inference 
```bash
python ./pytorch/run_inference.py 
or 
python ./pytorch/run_inference.py  --config_pth=../configs/nuscenes/all.pp.largea.config --ckpt=../checkpoint/voxelnet-140670.tckpt --input_folder=../input_point_clouds
```

### Performance in NuScenes validation set 
![test](https://gitlab.com/layssi/AI/raw/Mayank_Sati/PointPillars/images/eval.png)
![test2](https://gitlab.com/layssi/AI/raw/da365eb8364918b8c13c280fcefcac2d0ebf3c88/PointPillars/images/graph.jpg)

```
car Nusc dist AP@0.5, 1.0, 2.0, 4.0
62.90, 73.07, 76.77, 78.79
bicycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 0.00, 0.00, 0.00
bus Nusc dist AP@0.5, 1.0, 2.0, 4.0
9.53, 26.17, 38.01, 40.60
construction_vehicle Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.00, 0.00, 0.44, 1.43
motorcycle Nusc dist AP@0.5, 1.0, 2.0, 4.0
9.25, 12.90, 13.69, 14.11
pedestrian Nusc dist AP@0.5, 1.0, 2.0, 4.0
61.44, 62.61, 64.09, 66.35
traffic_cone Nusc dist AP@0.5, 1.0, 2.0, 4.0
11.63, 13.14, 15.81, 21.22
trailer Nusc dist AP@0.5, 1.0, 2.0, 4.0
0.80, 9.90, 17.61, 23.26
truck Nusc dist AP@0.5, 1.0, 2.0, 4.0
9.81, 21.40, 27.55, 30.34
```



## Try Kitti Viewer Web

### Major step

1. run ```python ./kittiviewer/backend/main.py main --port=xxxx``` in your server/local.

2. run ```cd ./kittiviewer/frontend && python -m http.server``` to launch a local web server.

3. open your browser and enter your frontend url (e.g. http://127.0.0.1:8000, default]).

4. input backend url (e.g. http://127.0.0.1:16666)

5. input root path, info path and det path (optional)

6. click load, loadDet (optional), input image index in center bottom of screen and press Enter.

### Inference step

Firstly the load button must be clicked and load successfully.

1. input checkpointPath and configPath.

2. click buildNet.

3. click inference.

![GuidePic](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/viewerweb.png)


## Try Kitti Viewer (Deprecated)

You should use kitti viewer based on pyqt and pyqtgraph to check data before training.

run ```python ./kittiviewer/viewer.py```, check following picture to use kitti viewer:
![GuidePic](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/simpleguide.png)

## Concepts


* Kitti lidar box

A kitti lidar box is consist of 7 elements: [x, y, z, w, l, h, rz], see figure.

![Kitti Box Image](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/kittibox.png)

All training and inference code use kitti box format. So we need to convert other format to KITTI format before training.

* Kitti camera box

A kitti camera box is consist of 7 elements: [x, y, z, l, h, w, ry].
