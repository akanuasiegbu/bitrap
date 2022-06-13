Repo contains the code for BiTrap adjusted for Pose predictions:

[BiPOCO: Bi-directional Trajectory Prediction with Pose Constraints for
Pedestrian Anomaly Detection]()

[BiTraP: Bi-directional Pedestrian Trajectory Prediction with Multi-modal Goal Estimation](https://arxiv.org/abs/2007.14558).



## Installation
### Dependencies
Our code was implemented using python and pytorch and tested on a desktop computer with Intel Xeon 2.10GHz CPU, NVIDIA TITAN X GPU and 128 GB memory.

* NVIDIA driver >= 418
* Python >= 3.6
* pytorch == 1.4.1 with GPU support (CUDA 10.1 & cuDNN 7)

Run following command to add bitrap path to the PYTHONPATH

  cd bidireaction-trajectory-prediction
  export PYTHONPATH=$PWD:PYTHONPATH

One can also use docker with `docker/Dockerfile`. 
# Pose Data Input into BiTRAP
* The inputted data into BiTrap for train and test poses can be found in this [folder]().
  * Next download the json files and put them in a folder. Then in config_for_my_data.py set ```loc['data_load']['avenue']['train_poses']```.   ```loc['data_load']['avenue']['test_poses']```,  ```loc['data_load']['st']['train_poses']```, and  ```loc['data_load']['st']['test_poses']``` to the correct directory.
* To recreate the pose input data
  * Run [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose/tree/ddaf4b99327132f7617a768a75f7cb94870ed57c) (commit number ddaf4b9)
    * Config file used was ```configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml```
    * Pretrained model used was ```pretrained_models/fast_res50_256x192.pth```
    * Tracker used was Human-ReID based tracking (```--pose_track```)
  * Next with json files from AlphaPose add the anomaly labels with the ```add_to_json_file.py``` for only the testing data

# Pretrained Models
Pretrained models for Avenue and ShanghaiTech can found [here]()


## Training
##### Pose trajectory training on Avenue and ShanghaiTech Dataset

Users can train the BiTraP models on Avenue and ShanghaiTech dataset easily by runing the following command:

Train on Avenue Dataset
```
python tools/train.py --config_file configs/avenue_pose_hc.yml
```

Train on ShanghaiTech Dataset
```
python  tools/train.py --config_file configs/st_pose_hc.yml
```

To train/inferece on CPU or GPU, simply add `DEVICE='cpu'` or  `DEVICE='cuda'`. By default we use GPU for both training and inferencing.

Note that you must set the input and output lengths to be the same in YML file used (```INPUT_LEN``` and ```PRED_LEN```) and ```config_for_my_data.py``` (```input_seq``` and ```pred_seq```)
## Inference 


##### Pose trajectory prediction on Avenue and ShanghaiTech Dataset
To predict the pose trajectory for first-person (ego-centric) view Avenue and ShanghaiTech datasets use commands below.

Test on Avenue dataset:
```
python tools/test.py --config_file configs/avenue_pose_hc.yml CKPT_DIR **DIR_TO_CKPT**

```

Test on ShanghaiTech dataset:
```
python tools/test.py --config_file configs/st_pose_hc.yml CKPT_DIR **DIR_TO_CKPT**
```

Note that you must set the input and output lengths to be the same in YML file used (```INPUT_LEN``` and ```PRED_LEN```) and ```config_for_my_data.py``` (```input_seq``` and ```pred_seq```)
## Citation

If you found the repo is useful, please feel free to cite our papers:
```
@article{yao2020bitrap,
  title={BiTraP: Bi-directional Pedestrian Trajectory Prediction with Multi-modal Goal Estimation},
  author={Yao, Yu and Atkins, Ella and Johnson-Roberson, Matthew and Vasudevan, Ram and Du, Xiaoxiao},
  journal={arXiv preprint arXiv:2007.14558},
  year={2020}
}
```
