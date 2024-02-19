#!/bin/bash

#Download caltech-256 dataset with pristine images
wget https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1
sudo apt install xz-utils
tar -xvf 256_ObjectCategories.tar?download=1
rm 256_ObjectCategories.tar?download=1
mkdir undistorted_datasets
mv ./256_ObjectCategories ./undistorted_datasets
mv ./undistorted_datasets/256_ObjectCategories ./undistorted_datasets/caltech256