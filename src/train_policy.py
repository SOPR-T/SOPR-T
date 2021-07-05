import argparse
import numpy as np
import os
import torch

import offline_agent
import online_agent
from utils.constants import env_list


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="HalfCheetah-v2")        # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)            # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--T", type=int, default=100)			  # epochs for offline algorithms and steps for online algorithms
	parser.add_argument("--save_interval", default=1, type=float) # For online algos, this means the intervals of saving steps
																  # and for offline algos, this means the intervals of saving epochs
	parser.add_argument("--batch_size", default=128, type=int)    # Mini batch size for networks
	parser.add_argument("--algo", type=str, default=None, 	 	  # select algos
						choices=['BCloning', 'BCQ', 'offDDPG', 'BEAR', 'CQL', 'onlineDDPG', 'onlineSAC', 'MOPO', 'CRR']) 
	parser.add_argument('--model_path', type=str, default='../models_try/')	# where to save model
	parser.add_argument('--buffer_path', type=str, default='D4RLdata')		# where is the data for offline training
	args = parser.parse_args()

	print("---------------------------------------")
	print(f'setting: training {args.algo}, Env: {args.env}, seed: {args.seed}')
	print("---------------------------------------")

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path, exist_ok=True)

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env_list[args.env]['state_dim'] #env.observation_space.shape[0]
	action_dim = env_list[args.env]['action_dim'] #env.action_space.shape[0] 
	max_action = env_list[args.env]['max_action'] #float(env.action_space.high[0])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	online = args.algo in ['onlineDDPG', 'onlineSAC']

	if online:
		trainer = online_agent.OnlineAgent(state_dim, action_dim, batch_size=args.batch_size, algo=args.algo, device=device)
	else:
		trainer = offline_agent.OfflineAgent(state_dim, action_dim, batch_size=args.batch_size, algo=args.algo, device=device)
	trainer.train(args, device=device)

	
	# os.makedirs(f'/mnt/exps/models/{args.model_path}/', exist_ok=True)
	os.system(f'mv {args.model_path}/{args.algo} /mnt/exps/models_our/{args.model_path}/')
	if args.algo == 'MOPO':
		os.system(f'mv {args.model_path}/MOPO_SAC/ /mnt/exps/models_retry/{args.model_path}/')
