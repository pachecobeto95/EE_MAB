import os, sys

DIR_PATH = os.path.dirname(__file__)

noise_list = [0, 0.5, 1, 1.5, 2, 2.5, 3]
blur_list = [0, 1, 2, 3, 4, 5]

distortion_level_dict = {"blur": blur_list, "noise": noise_list}


dataset_path = os.path.join(DIR_PATH, "undistorted_datasets", "caltech256")
distorted_dataset_path = os.path.join(DIR_PATH, "distorted_dataset", "caltech256")


dataset_name = "caltech256"
model_name = "mobilenet"
split_ratio = 0.2
batch_size_train = 64
batch_size_test = 1
seed = 42                     # the answer to life the universe and everything
use_gpu = True
exit_type = "bnpool"
distribution = "linear"
pretrained = True
max_patience = 20
weight_decay = 0.0005
