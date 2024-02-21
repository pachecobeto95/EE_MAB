import torchvision
import os, sys, time, math
from torchvision import transforms, utils, datasets
from PIL import Image
import torch, functools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchvision.models as models
from pthflops import count_ops
from torch import Tensor


class Early_Exit_DNN(nn.Module):
	def __init__(self, model_name: str, n_classes: int, 
		pretrained: bool, n_branches: int, input_dim: int, 
		exit_type: str, device, distribution="linear"):
	super(Early_Exit_DNN, self).__init__()

    """
    This class receives an DNN architecture name and inserts early-exit DNNs. 
    Args:

    model_name: model name 
    n_classes: number of classes in a classification problem, according to the dataset
    pretrained: 
    n_branches: number of branches (early exits) inserted into middle layers
    input_dim: dimension of the input image
    exit_type: type of the exits
    distribution: distribution method of the early exit blocks.
    device: indicates if the model will processed in the cpu or in gpu
    
    Note: the term "backbone model" refers to a regular DNN model, considering no early exits.

    """
    self.model_name = model_name
    self.n_classes = n_classes
    self.pretrained = pretrained
    self.n_branches = n_branches
    self.input_dim = input_dim
    self.exit_type = exit_type
    self.distribution = distribution
    self.device = device

    build_early_exit_dnn = self.select_dnn_model()
    build_early_exit_dnn()


	def select_dnn_model(self):
		"""
		This method selects the backbone to insert the early exits.
		"""

		architecture_dnn_model_dict = {"mobilenet": self.early_exit_mobilenet,
		"resnet18": self.early_exit_resnet18}

		return architecture_dnn_model_dict.get(self.model_name, self.invalid_model)

	def invalid_model(self):
		raise Exception("This DNN model has not implemented yet.")

	def is_suitable_for_exit(self):
		"""
		This method answers the following question. Is the position to place an early exit?
		"""

		intermediate_model = nn.Sequential(*(list(self.stages)+list(self.layers))).to(self.device)
		x = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)
		current_flop, _ = count_ops(intermediate_model, x, verbose=False, print_readable=False)
		return self.stage_id < self.n_branches and current_flop >= self.threshold_flop_list[self.stage_id]

	def add_exit_block(self):
		"""
		This method adds an early exit in the suitable position.
		"""
		input_tensor = torch.rand(1, 3, self.input_dim, self.input_dim)

		self.stages.append(nn.Sequential(*self.layers))
		x = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)
		feature_shape = nn.Sequential(*self.stages)(x).shape
		self.exits.append(EarlyExitBlock(feature_shape, self.pool_size, self.n_classes, self.exit_type, self.device))#.to(self.device))
		self.layers = nn.ModuleList()
		self.stage_id += 1    

		def set_device(self):
		"""
		This method sets the device that will run the DNN model.
		"""
		self.stages.to(self.device)
		self.exits.to(self.device)
		self.layers.to(self.device)
		self.classifier.to(self.device)

	def select_distribution_method(self):
		"""
		This method selects the distribution method to insert early exits into the middle layers.
		"""
		distribution_method_dict = {"linear":self.linear_distribution,
		"pareto":self.paretto_distribution,
		"fibonacci":self.fibo_distribution}
		
		return distribution_method_dict.get(self.distribution, self.invalid_distribution)

	def linear_distribution(self, i):
		"""
		This method defines the Flops to insert an early exits, according to a linear distribution.
		"""
		flop_margin = 1.0 / (self.n_branches+1)
		return self.total_flops * flop_margin * (i+1)

	def paretto_distribution(self, i):
		"""
		This method defines the Flops to insert an early exits, according to a pareto distribution.
		"""
		return self.total_flops * (1 - (0.8**(i+1)))

	def fibo_distribution(self, i):
		"""
		This method defines the Flops to insert an early exits, according to a fibonacci distribution.
		"""
		gold_rate = 1.61803398875
		return total_flops * (gold_rate**(i - self.num_ee))

	def verifies_nr_exits(self, backbone_model):
		"""
		This method verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
		"""
    
		total_layers = len(list(backbone_model.children()))
		if (self.n_branches >= total_layers):
			raise Exception("The number of early exits is greater than number of layers in the DNN backbone model.")


	def where_insert_early_exits(self):
		"""
		This method defines where insert the early exits, according to the dsitribution method selected.
		Args:

		total_flops: Flops of the backbone (full) DNN model.
		"""

		threshold_flop_list = []
		distribution_method = self.select_distribution_method()

		for i in range(self.n_branches):
			threshold_flop_list.append(distribution_method(i))

		return threshold_flop_list

	def invalid_model(self):
		raise Exception("This DNN backbone model has not implemented yet.")

	def invalid_distribution(self):
		raise Exception("This early-exit distribution has not implemented yet.")

	def countFlops(self, model):
		input_data = torch.rand(1, 3, self.input_dim, self.input_dim).to(self.device)
		flops, all_data = count_ops(model, input_data, print_readable=False, verbose=False)
		return flops


	def early_exit_mobilenet(self):
		"""
		This method inserts early exits into a Mobilenet V2 model
		"""

		self.stages = nn.ModuleList()
		self.exits = nn.ModuleList()
		self.layers = nn.ModuleList()
		self.cost = []
		self.stage_id = 0

		last_channel = 1280
    
		# Loads the backbone model. In other words, Mobilenet architecture provided by Pytorch.
		backbone_model = models.mobilenet_v2(self.pretrained).to(self.device)

		# It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
		self.verifies_nr_exits(backbone_model.features)
    
		# This obtains the flops total of the backbone model
		self.total_flops = self.countFlops(backbone_model)

		# This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
		self.threshold_flop_list = self.where_insert_early_exits()

		for i, layer in enumerate(backbone_model.features.children()):
      
			self.layers.append(layer)    
			if (self.is_suitable_for_exit()):
				self.add_exit_block()

		self.layers.append(nn.AdaptiveAvgPool2d(1))
		self.stages.append(nn.Sequential(*self.layers))
    
		self.classifier = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(last_channel, self.n_classes),)

		#self.set_device()
		self.softmax = nn.Softmax(dim=1)


	def forward_training(x):
		"""
		This method runs the DNN model during the training phase.
		x (tensor): input image
		"""

		conf_list, class_list, inf_time_list  = [], [], []
		starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

		cumulative_inf_time = 0.0

		for i, exitBlock in enumerate(self.exits):

			#This lines starts a timer to measure processing time
			starter.record()

			#This line process a DNN backbone until the (i+1)-th side branch (early-exit)
			x = self.stages[i](x)

			#This runs the early-exit classifications (prediction)
			output_branch = exitBlock(x)
			
			#This obtains the classification and confidence value in each side branch
			#Confidence is the maximum probability of belongs one of the predefined classes
			#The prediction , a.k.a inference_class,  is the argmax output. 
			conf_branch, prediction = torch.max(self.softmax(output_branch), 1)

			#This line terminates the timer started previously.
			ender.record()
			torch.cuda.synchronize()
			curr_time = starter.elapsed_time(ender)

			#This apprends the gathered confidences and classifications into a list
			conf_list.append(conf_branch.item()), class_list.append(prediction), inf_time_list.append(curr_time)

		#This measures the processing time for the last piece of DNN backbone
		starter.record()

		#This executes the last piece of DNN backbone
		x = self.stages[-1](x)

		x = torch.flatten(x, 1)

		#This generates the last-layer classification
		output = self.classifier(x)
		infered_conf, infered_class = torch.max(self.softmax(output), 1)

		#This ends the timer
		ender.record()
		torch.cuda.synchronize()
		curr_time = starter.elapsed_time(ender)


		conf_list.append(infered_conf), class_list.append(infered_class), inf_time_list.append(curr_time)

		cumulative_inf_time_list = np.cumsum(inf_time_list)

		return conf_list, class_list, inf_time_list, cumulative_inf_time_list


