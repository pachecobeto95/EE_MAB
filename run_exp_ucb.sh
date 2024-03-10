#!/bin/bash

#nohup python3 exp_ucb.py --n_branches 1 --n_rounds 5 --model_id 3 --loss_weights_type equal --distortion_type blur --reward_function adaee_basic2 --arm_selection_way ucb --n_iter 100000 &
#nohup python3 exp_ucb.py --n_branches 1 --n_rounds 5 --model_id 3 --loss_weights_type equal --distortion_type blur --reward_function adaee_basic2 --arm_selection_way random --n_iter 100000 &
#nohup python3 exp_ucb.py --n_branches 1 --n_rounds 5 --model_id 3 --loss_weights_type equal --distortion_type blur --reward_function adaee_basic2 --arm_selection_way fixed_threshold --fixed_threshold 0.6 --n_iter 100000 &
#nohup python3 exp_ucb.py --n_branches 1 --n_rounds 5 --model_id 3 --loss_weights_type equal --distortion_type blur --reward_function adaee_basic2 --arm_selection_way fixed_threshold --fixed_threshold 0.7 --n_iter 100000 &
#nohup python3 exp_ucb.py --n_branches 1 --n_rounds 5 --model_id 3 --loss_weights_type equal --distortion_type blur --reward_function adaee_basic2 --arm_selection_way fixed_threshold --fixed_threshold 0.9 --n_iter 100000 &


#nohup python3 exp_ucb.py --n_branches 1 --n_rounds 5 --model_id 3 --loss_weights_type equal --distortion_type blur --reward_function adaee_basic2 --arm_selection_way ucb --n_iter 100000 &

nohup python3 exp_ucb.py --n_branches 1 --n_rounds 5 --model_id 3 --loss_weights_type equal --distortion_type blur --reward_function alpha-fairness --arm_selection_way ucb --n_iter 100000 &
