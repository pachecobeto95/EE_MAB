import os, sys

DIR_PATH = os.getcwd()
noise_list = [0, 0.5, 1, 1.5, 2, 2.5, 3]
blur_list = [0, 1, 2, 3, 4, 5]

dataset_path = os.path.join(DIR_PATH, "undistorted_datasets", "256_ObjectCategories")
distorted_path = os.path.join(DIR_PATH, "distorted_dataset", "caltech256")
