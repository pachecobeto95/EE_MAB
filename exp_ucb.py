import argparse, logging, os, sys, config, utils
from tqdm import tqdm
import numpy as np
import pandas as pd
import ucb

def saveResults(results, resultPath):

	df = pd.DataFrame.from_dict(results)    
	# Append the DataFrame to the existing CSV file
	df.to_csv(resultPath, mode='a', header=not os.path.exists(resultPath))



def main(args):


	inf_data_dir_path = os.path.join(config.DIR_PATH, args.model_name, "inf_data",
		"inf_data_ee_%s_%s_branches_%s_id_%s.csv"%(args.model_name, args.n_branches, args.loss_weights_type, args.model_id))

	result_dir = os.path.join(config.DIR_PATH, args.model_name, "results")
	os.makedirs(result_dir, exist_ok=True)

	resultPath = os.path.join(result_dir, "results_ucb_ee_%s_%s_branches_%s_id_%s.csv"%(args.model_name, args.n_branches, args.loss_weights_type, args.model_id))
	performance_stats_path = os.path.join(result_dir, "perfomance_stats_ucb_ee_%s_%s_branches_%s_id_%s.csv"%(args.model_name, args.n_branches, args.loss_weights_type, args.model_id))

	distortion_level_list = config.distortion_level_dict[args.distortion_type]

	df = pd.read_csv(inf_data_dir_path)

	threshold_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	#overhead_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
	overhead_list = [0.5]
	
	context = {"distortion_type": args.distortion_type}

	for n_round in [0]:

		df = df.sample(frac=1).reset_index(drop=True)

		for overhead in overhead_list:

			for distortion_level in [0]:
				#print("Distortion Type: %s, Distortion Level: %s"%(args.distortion_type, distortion_level))
				print(f"Distortion Type: {args.distortion_type}, Distortion Level: {distortion_level}")

				context.update({"distortion_level": distortion_level})

				df_data = df[(df.distortion_type == args.distortion_type) & (df.distortion_level == distortion_level)]

				mab = ucb.UCB(threshold_list, args.c, args.n_iter, args.reward_function, overhead, args.arm_selection_way, 
					context)

				results, performance_stats = mab.adaee(df_data)

				saveResults(results, resultPath)
				saveResults(performance_stats, performance_stats_path)



if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Extract the confidences obtained by DNN inference for next experiments.")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256", "cifar10"], help='Dataset name.')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet
	parser.add_argument('--model_name', type=str, default=config.model_name, choices=["mobilenet", "alexnet"], 
		help='DNN model name (default: %s)'%(config.model_name))

	#parser.add_argument('--use_gpu', type=bool, default=config.use_gpu, help='Use GPU? Default: %s'%(config.use_gpu))

	parser.add_argument('--n_branches', type=int, help='Number of side branches.')

	parser.add_argument('--n_rounds', type=int, help='Number of rounds to run the experiment.')

	parser.add_argument('--model_id', type=int, help='Model_id.')

	parser.add_argument('--loss_weights_type', type=str, help='loss_weights_type.')

	parser.add_argument('--distortion_type', type=str, help='Distoriton Type applyed in dataset.',
		choices=list(config.distortion_level_dict.keys()))

	parser.add_argument('--n_iter', type=int, help='Number of iterations of UCB.')

	parser.add_argument('--c', type=int, help='Exploration and Exploitation ratio.')

	#parser.add_argument('--ucb_implementation', type=str, default="adaee", help='ucb_implementation.')

	parser.add_argument('--reward_function', type=str, help='Reward Function.')

	parser.add_argument('--arm_selection_way', type=str, help='Arm selection way.')


	args = parser.parse_args()

	main(args)