import config, utils
import argparse, logging, os

def main(args):

	models_dir_path = os.path.join(config.DIR_PATH, args.model_name, "models")
	history_dir_path = os.path.join(config.DIR_PATH, args.model_name, "history")

	os.makedirs(models_dir_path, exist_ok=True), os.makedirs(history_dir_path, exist_ok=True)

	model_save_path = os.path.join(models_dir_path, "ee_model_%s_%s_branches_id_%s.pth"%(args.model_name, args.n_branches, args.model_id))

	history_path = os.path.join(history_dir_path, "history_ee_model_%s_%s_branches_id_%s.csv"%(args.model_name, args.n_branches, args.model_id))

	indices_path = os.path.join(config.DIR_PATH, "indices_%s.pt"%(args.dataset_name))

	logPath = os.path.join(config.DIR_PATH, "log_train_ee_model_%s_%s_branches_id_%s.csv"%(args.model_name, args.n_branches, args.model_id))

	logging.basicConfig(level=logging.DEBUG, filename=logPath, filemode="a+", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

	device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

	train_loader, val_loader, test_loader = utils.load_caltech256(args, config.dataset_path, indices_path)



if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Training Early-exit DNN. These are the hyperparameters")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256", "cifar10"], help='Dataset name.')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet, ResNet18, ResNet152, VGG16
	parser.add_argument('--model_name', type=str, default=config.model_name, choices=["mobilenet", "alexnet"], 
		help='DNN model name (default: %s)'%(config.model_name))

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')

	#This argument defined the batch sizes. 
	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, 
		help='Train Batch Size. Default: %s'%(config.batch_size_train))

	parser.add_argument('--input_dim', type=int, default=330, help='Input Dim.')

	parser.add_argument('--dim', type=int, default=300, help='Image dimension')

	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')

	#parser.add_argument('--cuda', type=bool, default=config.use_gpu, help='Use GPU? Default: %s'%(config.use_gpu))

	parser.add_argument('--n_branches', type=int, help='Number of side branches.')

	parser.add_argument('--exit_type', type=str, default=config.exit_type, 
		help='Exit Type. Default: %s'%(config.exit_type))

	parser.add_argument('--distribution', type=str, default=config.distribution, 
		help='Distribution of the early exits. Default: %s'%(config.distribution))

	parser.add_argument('--pretrained', type=bool, default=config.pretrained, help='Backbone DNN is pretrained.')

	#parser.add_argument('--epochs', type=int, default=config.epochs, help='Epochs.')

	parser.add_argument('--max_patience', type=int, default=config.max_patience, help='Max Patience.')

	parser.add_argument('--model_id', type=int, help='Model_id.')

	#parser.add_argument('--loss_weights_type', type=str, help='loss_weights_type.')


	args = parser.parse_args()

	main(args)