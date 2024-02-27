import cv2, config
from torchvision import datasets, transforms
import torch, os, sys, requests
#import early_exit_dnn, b_mobilenet, ee_nn
import numpy as np
from torchvision.transforms.functional import InterpolationMode
#from torchvision.transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate


class ImageProcessor(object):
	def __init__(self, distortion_type, distortion_lvl):
		self.distortion_type = distortion_type
		self.distortion_lvl = distortion_lvl

	def blur(self):
		self.dist_img = cv2.GaussianBlur(self.image, (4*self.distortion_lvl+1, 4*self.distortion_lvl+1), 
			self.distortion_lvl)

	def noise(self):
		noise = np.random.normal(0, self.distortion_lvl, self.image.shape).astype(np.uint8)
		self.dist_img = cv2.add(self.image, noise)

	def distortion_not_found():
		raise ValueError("Invalid distortion type. Please choose 'blur' or 'noise'.")

	def apply(self, image_path):
		self.image = cv2.imread(image_path)
		dist_name = getattr(self, self.distortion_type, self.distortion_not_found)
		dist_name()

	def save_distorted_image(self, output_path):
		cv2.imwrite(output_path, self.dist_img)


def save_indices(train_idx, val_idx, test_idx, indices_path):

	data_dict = {"train": train_idx, "val": val_idx, "test": test_idx}
	torch.save(data_dict, indices_path)

def get_indices(dataset, split_ratio, indices_path):
	
	if (not os.path.exists(indices_path)):

		nr_samples = len(dataset)

		indices = list(torch.randperm(nr_samples).numpy())	

		train_val_size = nr_samples - int(np.floor(split_ratio * nr_samples))

		train_val_idx, test_idx = indices[:train_val_size], indices[train_val_size:]

		train_size = len(train_val_idx) - int(np.floor(split_ratio * len(train_val_idx) ))

		train_idx, val_idx = train_val_idx[:train_size], train_val_idx[train_size:]

		save_indices(train_idx, val_idx, test_idx, indices_path)

	else:
		data_dict = torch.load(indices_path)
		train_idx, val_idx, test_idx = data_dict["train"], data_dict["val"], data_dict["test"]	

	return train_idx, val_idx, test_idx


def load_caltech256(args, dataset_path, indices_path):

	#mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	mean, std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]

	torch.manual_seed(args.seed)

	interpolation = InterpolationMode(config.interpolation)

	transformations_train = transforms.Compose([
		transforms.Resize((args.input_dim, args.input_dim)),
		transforms.RandomChoice([
			transforms.ColorJitter(brightness=(0.80, 1.20)),
			transforms.RandomGrayscale(p = 0.25)]),
		transforms.CenterCrop((config.train_crop_size, config.train_crop_size)),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomRotation(25),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	transformations_test = transforms.Compose([
		transforms.Resize((args.input_dim, args.input_dim)),
		transforms.CenterCrop((args.dim, args.dim)),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])


	#transformations_train = transforms.Compose([
	#	transforms.RandomResizedCrop(config.train_crop_size, interpolation=interpolation, antialias=True),
	#	transforms.RandomHorizontalFlip(p=config.hflip_prob),
	#	transforms.TrivialAugmentWide(interpolation=interpolation),
	#	transforms.ToTensor(),
	#	transforms.Normalize(mean = mean, std = std),
	#	transforms.RandomErasing(p=config.random_erase)])

	#transformations_test = transforms.Compose([
	#	transforms.Resize((config.val_resize_size, config.val_resize_size)),
	#	transforms.CenterCrop((config.val_crop_size, config.val_crop_size)),
	#	transforms.ToTensor(), 
	#	transforms.Normalize(mean = mean, std = std),
	#	])

	#mixup_cutmix = get_mixup_cutmix(
	#	mixup_alpha=config.mixup_alpha, cutmix_alpha=config.cutmix_alpha, num_categories=257)


	# This block receives the dataset path and applies the transformation data. 
	train_set = datasets.ImageFolder(dataset_path, transform=transformations_train)
	val_set = datasets.ImageFolder(dataset_path, transform=transformations_test)
	test_set = datasets.ImageFolder(dataset_path, transform=transformations_test)

	train_idx, val_idx, test_idx = get_indices(train_set, args.split_ratio, indices_path)

	train_data = torch.utils.data.Subset(train_set, indices=train_idx)
	val_data = torch.utils.data.Subset(val_set, indices=val_idx)
	test_data = torch.utils.data.Subset(test_set, indices=test_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_train, 
		shuffle=True, num_workers=config.ngpus, pin_memory=True, collate_fn=default_collate)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=config.ngpus, 
		pin_memory=True, collate_fn=default_collate)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=config.ngpus, 
		pin_memory=True, collate_fn=default_collate)

	return train_loader, val_loader, test_loader
