import os, sys

DIR_PATH = os.path.dirname(__file__)

noise_list = [0, 0.5, 1, 1.5, 2, 2.5, 3]
blur_list = [0, 1, 2, 3, 4, 5]

distortion_level_dict = {"blur": blur_list, "noise": noise_list}


dataset_path = os.path.join(DIR_PATH, "undistorted_datasets", "caltech256")
distorted_dataset_path = os.path.join(DIR_PATH, "distorted_dataset", "caltech256")

n_class_dict = {"caltech256": 257}


dataset_name = "caltech256"
model_name = "mobilenet"
split_ratio = 0.1
#batch_size_train = 64
batch_size_train = 128
batch_size_test = 1
seed = 42                     # the answer to life the universe and everything
use_gpu = True
exit_type = "bnpool"
distribution = "linear"
pretrained = True
max_patience = 20
#weight_decay = 0.0005

weight_decay=2e-05 
ngpus=6
max_epochs=600 
momentum=0.9
lr=0.5
lr_min = 0
lr_scheduler='cosineannealinglr' 
lr_warmup_epochs=5
lr_warmup_method='linear' 
lr_warmup_decay=0.01
norm_weight_decay=0.0
label_smoothing=0.1 
mixup_alpha=0.2 
cutmix_alpha=1.0 
auto_augment='ta_wide'
random_erase=0.1
  
ra_sampler=True
ra_reps=4
ra_magnitude = 9

# EMA configuration
model_ema=True 
model_ema_steps=32 
model_ema_decay=0.99998

# Resizing
interpolation='bilinear'
val_resize_size=232 
val_crop_size=224
train_crop_size=176
hflip_prob = 0.5