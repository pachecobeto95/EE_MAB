import numpy as np
import config, logging


class UCB(object):
	def __init__(self, arms, c, n_rounds, reward_name, overhead, arm_selection_way, context, fixed_threshold, alpha=2):
		# Initialize the UCB class with required parameters.
		# Arms: List of arms (thresholds) for the arms in the context of multi-armed bandit problem.
		# c: Exploration parameter for UCB algorithm.
		# n_rounds: Total number of rounds for the algorithm to run.
		# ucb_implementation: Implementation method for UCB algorithm (not clear from the provided code).
		# reward_name: Name of the reward function to be used.
		# costs: Costs associated with the arms or actions.
		# overhead: Overhead associated with actions or decisions.
		# arm_selection_way: Method for selecting arms based on certain criteria.
		# alpha: Parameter used in reward calculations.

		# Initialize instance variables.
		self.arms = arms
		self.n_arms = len(arms)
		self.n_rounds = n_rounds
		self.total_rewards = np.zeros(self.n_arms)  # Total rewards accumulated for each arm.
		self.n_pulls = np.zeros(self.n_arms) # Number of times each arm is pulled.
		self.total_pulls = 0# Total number of pulls across all arms.
		#self.ucb_implementation = ucb_implementation  # UCB implementation method.
		self.reward_name = reward_name  # Name of the reward function.
		#self.costs = costs  # Costs associated with each arm.
		self.overhead = overhead# Overhead associated with decisions.
		self.c = c  # Exploration-exploitation parameter for UCB.
		self.cumulative_regret = 0  # Cumulative regret over all rounds.
		self.cumulative_regret_list = np.zeros(n_rounds)  # List to store cumulative regret for each round.
		self.inst_regret_list = np.zeros(n_rounds)# List to store instantaneous regret for each round.
		self.selected_arm_list = np.zeros(n_rounds)   # List to store the selected arm for each round.
		self.correct_list = np.zeros(n_rounds)# List to store correctness information for each round.
		self.offloading_list = np.zeros(n_rounds) # List to store offloading decisions for each round.
		self.arm_selection_way = arm_selection_way# Method for selecting arms.
		self.alpha = alpha# Alpha parameter for reward calculations.
		self.total_offloading = 0 # Total offloading decisions made.
		self.context = context
		self.inf_time_list = np.zeros(n_rounds)
		self.fixed_threshold = fixed_threshold


	# Function to randomly select an input from the provided dataframe.
	def pick_random_input(self, df):
		idx = np.random.choice(range(df.shape[0]))
		row = df.iloc[[idx]]
		return row

	# Function to extract information related to inference data.
	def get_inf_data(self, row, threshold):
		conf_branch, conf_final = row.conf_branch_1.item(), row.conf_branch_2.item()
		delta_conf = conf_final - conf_branch if(conf_final >= threshold) else max([conf_branch, conf_final]) - conf_branch
		return conf_branch, conf_final, delta_conf

	# Reward function: AdaEE basic.
	def reward_adaee_basic(self, arm, row):
		threshold = self.arms[arm]
		conf_branch, conf_final, delta_conf = self.get_inf_data(row, threshold)
		return max(delta_conf, 0) - self.overhead if (conf_branch < threshold) else 0

	# Reward function: AdaEE basic 2.
	def reward_adaee_basic_2(self, arm, row):
		threshold = self.arms[arm]
		conf_branch, conf_final, delta_conf = self.get_inf_data(row, threshold)
		return delta_conf - self.overhead if (conf_branch < threshold) else 0

	# Reward function: Alpha fairness.
	def reward_alpha_fairness(self, arm, row):
		threshold = self.arms[arm]
		_, _, delta_conf = self.get_inf_data(row, threshold)
		r = max(delta_conf, 0) - self.overhead
		if(self.alpha == 1):
			return np.log(r)
		else:
			return (r**(1-self.alpha) - 1) / (1 - self.alpha)


	def reward_i_split_ee(self, arm, row):
		conf_branch, conf_final, delta_conf = self.get_inf_data(row)
		gamma, mu = 1/10, 1
		threshold = self.arms[arm]

		if(conf_branch >= threshold and self.n_arms <= 5):
			reward = confidence - mu * gamma * (self.n_arms+1+(2*(self.n_arms + 1)))

		elif(n_exit <= 5 and conf_branch < threshold):
			#cost = o + gamma * (n_exit+1+(2*(n_exit + 1)))
			reward = conf_final - mu * self.overhead - mu * gamma * (self.n_arms+1+(2*(self.n_arms + 1)))

	# Function to compute reward based on the selected reward function.
	def compute_reward(self, arm, input_data):
		return getattr(self, "reward_%s" % (self.reward_name))(arm, input_data)

	# Function to check whether offloading is needed based on the selected arm and input data.
	def check_offloading(self, arm, row):
		threshold = self.arms[arm]
		conf_branch, conf_final, delta_conf = self.get_inf_data(row, threshold)
		return 1 if (conf_branch < threshold) else 0

	# Function to pull a specific arm based on input data.
	def pull_arm(self, arm, input_data):
		reward = self.compute_reward(arm, input_data)
		#self.offloading_flag = self.check_offloading(arm, input_data)
		#self.total_offloading += self.offloading_flag
		self.total_rewards[arm] += reward
		self.n_pulls[arm] += 1
		self.total_pulls += 1
		return reward

	# Function to pull the optimal arm based on input data.
	def pull_optimal_arm(self, arm, input_data):
		threshold = self.arms[arm]
		_, _, delta_conf = self.get_inf_data(input_data, threshold)
		return max(0, delta_conf - self.overhead)

	# Function to compute the Upper Confidence Bound (UCB) for a given arm.
	def compute_ucb(self, arm):
		return self.total_rewards[arm] / self.n_pulls[arm] + (self.c * np.sqrt(np.log(self.total_pulls) / self.n_pulls[arm]))

	# Function to select an arm using UCB strategy.
	def ucb_selection(self, n_round):
		ucb_values = [self.compute_ucb(arm) for arm in range(self.n_arms)]
		selected_arm = np.argmax(ucb_values)
		threshold = self.arms[selected_arm]
		self.selected_arm_list[n_round] = round(threshold, 2)
		return selected_arm

	# Function for random arm selection.
	def random_selection(self, n_round):
		selected_arm = np.random.choice(range(self.n_arms))
		threshold = self.arms[selected_arm]
		self.selected_arm_list[n_round] = round(threshold, 2)
		return selected_arm

	# Function for fixed threshold-based arm selection.
	def fixed_threshold_selection(self, n_round):
		#arm_list = list(range(self.n_arms))
		self.selected_arm_list[n_round] = round(self.fixed_threshold, 2)
		return self.arms.index(self.fixed_threshold)

	# Function for last layer arm selection (not implemented in the provided code).
	def last_layer_selection(self, n_round):
		self.selected_arm_list[n_round] = round(self.arms[-1], 2)		
		return -1

	# Function to select arm based on the specified method.
	def select_arm(self, n_round):
		return getattr(self, "%s_selection" % (self.arm_selection_way))(n_round)

	# Function to compute regret for a given round.
	def compute_regret(self, reward, optimal_reward, n_round):
		inst_regret = optimal_reward - reward
		self.cumulative_regret += inst_regret
		self.cumulative_regret_list[n_round] = self.cumulative_regret
		self.inst_regret_list[n_round] = round(inst_regret, 5)

	# Function to check correctness of decision made and the inference time.
	def compute_stats(self, row, arm, n_round):
		conf_branch, conf_final = row.conf_branch_1.item(), row.conf_branch_2.item()
		threshold = self.arms[arm]

		if(conf_branch >= threshold):
			correct = row.correct_branch_1.item()
			inference_time = row.cum_inf_time_branch_1.item()
			offloading_flag = 0
		else:
			inference_time = row.cum_inf_time_branch_2.item()
			offloading_flag = 1

			if(conf_final >= threshold):
				correct = row.correct_branch_2.item()

			else:
				conf_branches = [conf_branch, conf_final]
				correct_branches = [row.correct_branch_1.item(), row.correct_branch_2.item()]
				correct = correct_branches[np.argmax(conf_branches)]
					


		self.correct_list[n_round] = correct
		self.inf_time_list[n_round] = inference_time
		self.offloading_list[n_round] = offloading_flag

	# Main function implementing the AdaEE algorithm.
	def adaee(self, df_data):
	# Pull each arm once initially.
		
		for arm_to_pull in range(self.n_arms):
			random_input = self.pick_random_input(df_data)
			reward = self.pull_arm(arm_to_pull, random_input)

		# Start pulling arms based on selection strategy.
		for n_round in range(self.n_arms, self.n_rounds):

			random_input = self.pick_random_input(df_data)
			arm_to_pull = self.select_arm(n_round)
			reward = self.pull_arm(arm_to_pull, random_input)
			optimal_reward = self.pull_optimal_arm(arm_to_pull, random_input)
			self.compute_regret(reward, optimal_reward, n_round)
			self.compute_stats(random_input, arm_to_pull, n_round)
			#self.compute_inference_time(random_input, arm_to_pull, n_round)
			#self.offloading_list[n_round] = self.offloading_flag

		# Calculate accuracy and offloading probability.
		acc = sum(self.correct_list) / len(self.correct_list)
		avg_inf_time, std_inf_time = np.mean(self.inf_time_list), np.std(self.inf_time_list)
		offloading_prob = self.total_offloading / self.n_rounds

		item, contagem = np.unique(self.selected_arm_list, return_counts=True)

		logging.info("Acc: %s, Inf Time: %s, Most selected arm: %s"%(acc, avg_inf_time, item[np.argmax()]))
		#print("Most selected Arm: %s"%(item[np.argmax()]))

		# Gather performance results.
		performance_results = {"acc": [acc], "overhead": [self.overhead], "c": [self.c], "offloading_prob": [offloading_prob],
		"distortion_type": [self.context["distortion_type"]], "distortion_level": [self.context["distortion_level"]],
		"arm_selection_way": [self.arm_selection_way], "reward_function": [self.reward_name],
		"avg_inf_time": avg_inf_time, "std_inf_time": std_inf_time}

		# Gather other results for analysis.
		results = {"selected_arm": self.selected_arm_list, "regret": self.inst_regret_list,
		"overhead": [round(self.overhead, 2)] * self.n_rounds,
		"cumulative_regret": self.cumulative_regret_list, "c": [self.c] * self.n_rounds,
		"distortion_type": self.n_rounds*[self.context["distortion_type"]],
		"distortion_level": self.n_rounds*[self.context["distortion_level"]],
		"arm_selection_way": self.n_rounds*[self.arm_selection_way], "reward_function": self.n_rounds*[self.reward_name],
		"inf_time": self.inf_time_list}

		return results, performance_results


