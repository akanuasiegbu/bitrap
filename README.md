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
The inputted data into BiTrap for Poses can be found in this folder.
Next download the json files and put them in a folder. Then in config_for_my_data.py set loc[‘data_load’][‘avenue’][‘train_poses’].  Loc[‘data_load’][‘avenue’][‘tet_poses’],  loc[‘data_load’][st][‘train_poses’], and  loc[‘data_load’][st][‘test_poses’] to the correct directory.
To recreate the input data
Run AlphaPose (commit number xxx)
Put how you run AlphaPose details here
Next with json files from AlphaPose we need to add the anomaly labels with the add_to_json_file.py.


## Training
Users can train the BiTraP models on Avenue and ShanghaiTech dataset easily by runing the following command:
```
python tools/train.py --config_file **DIR_TO_THE_YML_FILE** 
```


To train/inferece on CPU or GPU, simply add `DEVICE='cpu'` or  `DEVICE='cuda'`. By default we use GPU for both training and inferencing.

## Inference 


### Bounding box trajectory prediction on Avenue and Shanghai Tech
We predict the bounding box coordinate trajectory for first-person (ego-centric) view Avenue and ShanghaiTech datasets.
Test on Avenue dataset:
```
python tools/test.py --config_file configs/avenue_pose_hc.yml CKPT_DIR **DIR_TO_CKPT**

```

Test on ShanghaiTech dataset:
```
python tools/test.py --config_file configs/st_pose_hc.yml CKPT_DIR **DIR_TO_CKPT**
```

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
