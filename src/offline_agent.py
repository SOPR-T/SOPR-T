import numpy as np
import d3rlpy
import os
from d3rlpy.dataset import MDPDataset
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer
from d3rlpy.metrics.scorer import dynamics_reward_prediction_error_scorer
from d3rlpy.metrics.scorer import dynamics_prediction_variance_scorer
import torch
from utils.getD4RLData import getD4RLData

# wrapper of d3rlpy.algo
class OfflineAgent:
    def __init__(self, state_dim, action_dim, batch_size=256, algo='bear', device=torch.device('cuda')) -> None:
        self.device = device
        use_gpu = (device == torch.device('cuda'))
        self.algo_name = algo
        if algo == 'BEAR':
            self.algo = d3rlpy.algos.BEAR(use_gpu=use_gpu, batch_size=batch_size)
        elif algo == 'CQL':
            self.algo = d3rlpy.algos.CQL(use_gpu=use_gpu, batch_size=batch_size)
        elif algo == 'BCloning':
            self.algo = d3rlpy.algos.BC(use_gpu=use_gpu, batch_size=batch_size)
        elif algo == 'BCQ':
            self.algo = d3rlpy.algos.BCQ(use_gpu=use_gpu, batch_size=batch_size)
        elif algo == 'offDDPG':
            self.algo = d3rlpy.algos.DDPG(use_gpu=use_gpu, batch_size=batch_size)
        elif algo == 'MOPO':
            # self.algo = d3rlpy.algos.SAC(use_gpu=use_gpu, batch_size=batch_size)
            self.dynamic = d3rlpy.algos.MOPO(use_gpu=use_gpu, batch_size=batch_size)
            self.algo = d3rlpy.algos.SAC(use_gpu=use_gpu, batch_size=batch_size)
            self.algo_name = "MOPO_SAC"
        elif algo == 'CRR':
            self.algo = d3rlpy.algos.CRR(use_gpu=use_gpu, batch_size=batch_size)

        observations = np.random.random((10, state_dim))
        actions = np.random.random((10, action_dim))
        rewards = np.random.random(10)
        terminals = np.random.randint(2, size=10)
        terminals[5] = 1
        terminals[2] = 0
        dataset = MDPDataset(observations, actions, rewards, terminals)
        self.algo.build_with_dataset(dataset)
        
    # select action for states
    def select_action_buffer(self, state):
        return torch.tensor(self.algo.predict(state.cpu().numpy())).cuda()

    # get total probability of action for state
    def get_action_buffer(self, state: torch.tensor, action: torch.tensor):
        assert self.algo_name != 'BCloning'
        state = state.to(self.device)
        action = action.to(self.device)
        dist = self.algo._impl._policy.dist(state)
        return dist.log_prob(action).sum(1)    # (batch,), log_prob

    # get seperate action log probability for states
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
        o, a, r, d = getD4RLData(args.buffer_path)
        dataset = d3rlpy.dataset.MDPDataset(o, a, r, d)

        if self.algo_name == 'MOPO_SAC':
            self.dynamic.fit(
                dataset.episodes,
                eval_episodes=dataset.episodes,
                n_epochs=100,
                save_interval=args.save_interval,
                scorers={
                    'observation_error': dynamics_observation_prediction_error_scorer,
                    'reward_error': dynamics_reward_prediction_error_scorer,
                    'variance': dynamics_prediction_variance_scorer,
                },
                logdir=args.model_path,
                experiment_name='MOPO',
                with_timestamp=False
            )
            self.dynamic.set_params(n_transitions=400, horizon=5, lam=1.0)
            self.algo = d3rlpy.algos.SAC(use_gpu=(device==torch.device('cuda')), batch_size=args.batch_size, generator=self.dynamic)
        
        self.algo.fit(
            dataset.episodes,
            n_epochs=args.T,
            save_interval=args.save_interval,
            logdir=args.model_path,
            with_timestamp=False,
            experiment_name=self.algo_name)