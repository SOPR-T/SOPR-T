import copy
import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from ...augmentation import AugmentationPipeline
from ...gpu import Device
from ...models.builders import (
    create_categorical_policy,
    create_discrete_q_function,
    create_parameter,
    create_squashed_normal_policy,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import (
    CategoricalPolicy,
    EnsembleDiscreteQFunction,
    Parameter,
    SquashedNormalPolicy,
)
from ...preprocessing import ActionScaler, Scaler
from ...torch_utility import augmentation_api, hard_sync, torch_api, train_api
from .base import TorchImplBase
from .ddpg_impl import DDPGBaseImpl
from .utility import DiscreteQFunctionMixin


class SACImpl(DDPGBaseImpl):

    _policy: Optional[SquashedNormalPolicy]
    _targ_policy: Optional[SquashedNormalPolicy]
    _temp_learning_rate: float
    _temp_optim_factory: OptimizerFactory
    _initial_temperature: float
    _log_temp: Optional[Parameter]
    _temp_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        target_reduction_type: str,
        initial_temperature: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        augmentation: AugmentationPipeline,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            target_reduction_type=target_reduction_type,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            augmentation=augmentation,
        )
        self._temp_learning_rate = temp_learning_rate
        self._temp_optim_factory = temp_optim_factory
        self._initial_temperature = initial_temperature

        # initialized in build
        self._log_temp = None
        self._temp_optim = None

    def build(self) -> None:
        self._build_temperature()
        super().build()
        self._build_temperature_optim()

    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def _build_temperature(self) -> None:
        initial_val = math.log(self._initial_temperature)
        self._log_temp = create_parameter((1, 1), initial_val)

    def _build_temperature_optim(self) -> None:
        assert self._log_temp is not None
        self._temp_optim = self._temp_optim_factory.create(
            self._log_temp.parameters(), lr=self._temp_learning_rate
        )

    def _compute_actor_loss(self, obs_t: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._q_func is not None
        action, log_prob = self._policy.sample_with_log_prob(obs_t)
        entropy = self._log_temp().exp() * log_prob
        q_t = self._q_func(obs_t, action, "min")
        return (entropy - q_t).mean()

    @train_api
    @torch_api(scaler_targets=["obs_t"])
    def update_temp(self, obs_t: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        assert self._temp_optim is not None
        assert self._policy is not None
        assert self._log_temp is not None

        self._temp_optim.zero_grad()

        with torch.no_grad():
            _, log_prob = self._policy.sample_with_log_prob(obs_t)
            targ_temp = log_prob - self._action_size

        loss = -(self._log_temp().exp() * targ_temp).mean()

        loss.backward()
        self._temp_optim.step()

        # current temperature value
        cur_temp = self._log_temp().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_temp

    @augmentation_api(targets=["x"])
    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action, log_prob = self._policy.sample_with_log_prob(x)
            entropy = self._log_temp().exp() * log_prob
            target = self._targ_q_func.compute_target(
                x, action, reduction=self._target_reduction_type
            )
            if self._target_reduction_type == "none":
                return target - entropy.view(1, -1, 1)
            else:
                return target - entropy


class DiscreteSACImpl(DiscreteQFunctionMixin, TorchImplBase):

    _actor_learning_rate: float
    _critic_learning_rate: float
    _temp_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _temp_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _n_critics: int
    _initial_temperature: float
    _use_gpu: Optional[Device]
    _policy: Optional[CategoricalPolicy]
    _q_func: Optional[EnsembleDiscreteQFunction]
    _targ_q_func: Optional[EnsembleDiscreteQFunction]
    _log_temp: Optional[Parameter]
    _actor_optim: Optional[Optimizer]
    _critic_optim: Optional[Optimizer]
    _temp_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        initial_temperature: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        augmentation: AugmentationPipeline,
    ):
        super().__init__(
            observation_shape, action_size, scaler, None, augmentation
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._temp_learning_rate = temp_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._temp_optim_factory = temp_optim_factory
        self._actor_encoder_factory = actor_encoder_factory
        self._critic_encoder_factory = critic_encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._n_critics = n_critics
        self._initial_temperature = initial_temperature
        self._use_gpu = use_gpu

        # initialized in build
        self._q_func = None
        self._policy = None
        self._targ_q_func = None
        self._log_temp = None
        self._actor_optim = None
        self._critic_optim = None
        self._temp_optim = None

    def build(self) -> None:
        self._build_critic()
        self._build_actor()
        self._build_temperature()

        # setup target networks
        self._targ_q_func = copy.deepcopy(self._q_func)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_critic_optim()
        self._build_actor_optim()
        self._build_temperature_optim()

    def _build_critic(self) -> None:
        self._q_func = create_discrete_q_function(
            self._observation_shape,
            self._action_size,
            self._critic_encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        self._critic_optim = self._critic_optim_factory.create(
            self._q_func.parameters(), lr=self._critic_learning_rate
        )

    def _build_actor(self) -> None:
        self._policy = create_categorical_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )

    def _build_actor_optim(self) -> None:
        assert self._policy is not None
        self._actor_optim = self._actor_optim_factory.create(
            self._policy.parameters(), lr=self._actor_learning_rate
        )

    def _build_temperature(self) -> None:
        initial_val = math.log(self._initial_temperature)
        self._log_temp = create_parameter((1, 1), initial_val)

    def _build_temperature_optim(self) -> None:
        assert self._log_temp is not None
        self._temp_optim = self._temp_optim_factory.create(
            self._log_temp.parameters(), lr=self._temp_learning_rate
        )

    @train_api
    @torch_api(scaler_targets=["obs_t", "obs_tpn"])
    def update_critic(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        obs_tpn: torch.Tensor,
        ter_tpn: torch.Tensor,
        n_steps: torch.Tensor,
        masks: torch.Tensor,
    ) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(obs_tpn)
        q_tpn *= 1.0 - ter_tpn

        loss = self.compute_critic_loss(
            obs_t, act_t.long(), rew_tpn, q_tpn, n_steps, masks
        )

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    @augmentation_api(targets=["x"])
    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        assert self._log_temp is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            log_probs = self._policy.log_probs(x)
            probs = log_probs.exp()
            entropy = self._log_temp().exp() * log_probs
            target = self._targ_q_func.compute_target(x)
            keepdims = True
            if target.dim() == 3:
                entropy = entropy.unsqueeze(-1)
                probs = probs.unsqueeze(-1)
                keepdims = False
            return (probs * (target - entropy)).sum(dim=1, keepdim=keepdims)

    @augmentation_api(targets=["obs_t"])
    def compute_critic_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        n_steps: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        return self._compute_critic_loss(
            obs_t, act_t, rew_tpn, q_tpn, n_steps, masks
        )

    def _compute_critic_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        n_steps: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            obs_t, act_t, rew_tpn, q_tpn, self._gamma ** n_steps, masks=masks
        )

    @train_api
    @torch_api(scaler_targets=["obs_t"])
    def update_actor(self, obs_t: torch.Tensor) -> np.ndarray:
        assert self._q_func is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(obs_t)

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    @augmentation_api(targets=["obs_t"])
    def compute_actor_loss(self, obs_t: torch.Tensor) -> torch.Tensor:
        return self._compute_actor_loss(obs_t)

    def _compute_actor_loss(self, obs_t: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None
        assert self._policy is not None
        assert self._log_temp is not None
        with torch.no_grad():
            q_t = self._q_func(obs_t, reduction="min")
        log_probs = self._policy.log_probs(obs_t)
        probs = log_probs.exp()
        entropy = self._log_temp().exp() * log_probs
        return (probs * (entropy - q_t)).sum(dim=1).mean()

    @train_api
    @torch_api(scaler_targets=["obs_t"])
    def update_temp(self, obs_t: torch.Tensor) -> np.ndarray:
        assert self._temp_optim is not None
        assert self._policy is not None
        assert self._log_temp is not None

        self._temp_optim.zero_grad()

        with torch.no_grad():
            log_probs = self._policy.log_probs(obs_t)
            probs = log_probs.exp()
            expct_log_probs = (probs * log_probs).sum(dim=1, keepdim=True)
            entropy_target = 0.98 * (-math.log(1 / self.action_size))
            targ_temp = expct_log_probs + entropy_target

        loss = -(self._log_temp().exp() * targ_temp).mean()

        loss.backward()
        self._temp_optim.step()

        # current temperature value
        cur_temp = self._log_temp().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_temp

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.best_action(x)

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.sample(x)

    def update_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        hard_sync(self._targ_q_func, self._q_func)
