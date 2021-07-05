# Supervised Off-Policy Ranking

This repository is the official implementation of Supervised Off-Policy Ranking. 

## Requirements

To install requirements:

```setup
cd d3rlpy-master
pip install -r requirements.txt
pip install -e .
```

The implementation of policy models used in this work is based on the d3rlpy repository https://github.com/takuseno/d3rlpy. We used a copy of it and made a tiny change on it. You can find the change in line 255 in `d3rlpy-master/d3rlpy/models/torch/policies.py`.


## Dataset

We evaluate SOPR on D4RL dataset https://github.com/rail-berkeley/d4rl, which is commonly used in offline RL studies.
For the dataset corresponding to Gym-MuJoCo_v2 used in our work, you can also directly download in http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2.


## Policy models

The training policies, validation policies and test policies can be found in 
- [Policy models](https://drive.google.com/file/d/1yPrnvyJNK4zmVyvFdtUI8bVOtZxO0qhN/view?usp=sharing). 


## Training

To train SOPR-T, you can run this command:

```train_random
cd src
python train_ranknet.py --env HalfCheetah-v2 \
                        --seed 0 \
                        --cluster_path /path/to/state_clusters/ \
                        --model_path /path/to/training_set/ \
                        --validation_path /path/to/validation_set/ \
                        --logdir log/ \
                        --save_dir ranknet/ \
                        --data_type expert \
                        --training_mode random \
                        --validation_mode random \
                        --score_mode mean \
                        --validation_repeat 3 \
                        --epochs 200
```

or use the following command to train with less data (e.g. 16k) and without mini-batch training.

```train_random
cd src
python train_ranknet.py --env HalfCheetah-v2 \
                        --seed 0 \
                        --cluster_path /path/to/state_clusters/ \
                        --model_path /path/to/training_set/ \
                        --validation_path /path/to/validation_set/ \
                        --logdir log/ \
                        --save_dir ranknet/ \
                        --data_type expert \
                        --training_mode fixed \
                        --training_states 16k \
                        --validation_mode fixed \
                        --validation_states 16k \
                        --score_mode mean \
                        --epochs 200
```
The training scripts will automatically save three SOPR-T models corresponding to the highest validation rank correlation, the lowest validation loss, and the initial one, respectively. 


## Evaluation

To evaluate the trained SOPR-T model on the test policy set, you can run the following command:

```eval
cd src
python test_ranknet.py --env HalfCheetah-v2 \
                       --seed 0 \
                       --seed1 0 \
                       --ranking_model /path/to/ranknet/ \
                       --ranknet_type corr \
                       --cluster_path /path/to/state_cluster/ \
                       --model_path1 /path/to/test_set1/ \
                       --model_path2 /path/to/test_set2/ \
                       --data_type expert \
                       --eval_mode fixed \
                       --test_states 16k \
                       --final_score mean \
                       --score_mode mean \
                       --save_dir results
```

The results will be automatically saved at save_dir as .txt files.


## SOPR-T Models 

You can download our trained SOPR-T models here:

- [SOPR-T models](https://drive.google.com/drive/folders/18QNVhh3Fv8FXrdC-sr-TdI_SHTF4Si00?usp=sharing) 




