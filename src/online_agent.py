import numpy as np
import d3rlpy
import os
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
import gym
import torch


# wrapper of d3rlpy.algo
class OnlineAgent:
    def __init__(self, state_dim, action_dim, batch_size=256, algo='bear', device=torch.device('cuda')) -> None:
        use_gpu = (device == torch.device('cuda'))
        self.algo_name = algo
        if algo == 'onlineSAC':
            self.algo = d3rlpy.algos.SAC(use_gpu=use_gpu, batch_size=batch_size)
        elif algo == 'onlineDDPG':
            self.algo = d3rlpy.algos.DDPG(use_gpu=use_gpu, batch_size=batch_size)

        observations = np.random.random((10, state_dim))
        actions = np.random.random((10, action_dim))
        rewards = np.random.random(10)
        terminals = np.random.randint(2, size=10)
        terminals[5] = 1
        terminals[2] = 0
        dataset = MDPDataset(observations, actions, rewards, terminals)
        self.algo.build_with_dataset(dataset)
        
    # select action for state buffer
    def select_action_buffer(self, state):
        return torch.tensor(self.algo.predict(state.cpu().numpy())).cuda()

    # get aggregated log probability of action for state
    def get_action_buffer(self, state: torch.tensor, action: torch.tensor):
        assert self.algo_name != 'BCloning'
        state = state.to(self.device)
        action = action.to(self.device)
        dist = self.algo._impl._policy.dist(state)
        return dist.log_prob(action).sum(1)    # (batch,), log_prob

    # get seperate log probability of each dim of action for state
    def get_action_buffer_seperate(self, state: torch.tensor, action: torch.tensor):
        assert self.algo_name != 'BCloning'
        state = state.to(self.device)
        action = action.to(self.device)
        dist = self.algo._impl._policy.dist(state)
        return dist.log_prob(action)    # (batch, action), log_prob

    # load policy
    def load(self, path):
        self.algo.load_model(path)

    def train(self, args, device):
        
        env=gym.make(args.env)
        eval_env = gym.make(args.env)
        self.algo.fit_online(
            env,
            n_steps=args.T,
            eval_env=eval_env,
            with_timestamp=False,
            logdir=args.model_path,
            experiment_name=self.algo_name,
            save_interval=1,
            n_steps_per_epoch=args.save_interval
        )