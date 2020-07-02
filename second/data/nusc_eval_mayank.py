import fire 

from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval

def eval_main(root_path, version, eval_version, res_path, eval_set, output_dir):
    nusc = NuScenes(
        version=version, dataroot=str(root_path), verbose=False)

    cfg = config_factory(eval_version)
    nusc_eval = NuScenesEval(nusc, config=cfg, result_path=res_path, eval_set=eval_set, 
                            output_dir=output_dir,
                            verbose=False)
    nusc_eval.main(render_curves=False)

if __name__ == "__main__":

    # fire.Fire(eval_main)

 #    -res_path = "/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/mayank_pc_trained/eval_results/step_0/results_nusc.json" \
 #                - -eval_set = mini_train -\
 # -output_dir = "/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/mayank_pc_trained/eval_results/step_0"'' \
 #               ' returned non-zero exit status 1.

    # python /home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/data/nusc_eval.py
    root_path="/home/mayank_sati/Documents/point_clouds/nuscene_v1.0-mini"
    version='v1.0-mini'
    eval_version='cvpr_2019'
    res_path="/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/mayank_pc_trained/eval_results/step_20/results_nusc.json"
    eval_set='mini_train'
    output_dir="/home/mayank_sati/pycharm_projects/pytorch/second.pytorch_traveller59_date_9_05/second/mayank_pc_trained/eval_results/step_20"
    eval_main(root_path, version, eval_version, res_path, eval_set, output_dir)