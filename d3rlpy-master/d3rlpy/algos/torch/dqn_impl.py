import copy
from typing import Optional, Sequence

import numpy as np
import torch
from torch.optim import Optimizer

from ...augmentation import AugmentationPipeline
from ...gpu import Device
from ...models.builders import create_discrete_q_function
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import EnsembleDiscreteQFunction
from ...preprocessing import Scaler
from ...torch_utility import augmentation_api, hard_sync, torch_api, train_api
from .base import TorchImplBase
from .utility import DiscreteQFunctionMixin


class DQNImpl(DiscreteQFunctionMixin, TorchImplBase):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _n_critics: int
    _target_reduction_type: str
    _use_gpu: Optional[Device]
    _q_func: Optional[EnsembleDiscreteQFunction]
    _targ_q_func: Optional[EnsembleDiscreteQFunction]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        target_reduction_type: str,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        augmentation: AugmentationPipeline,
    ):
        super().__init__(
            observation_shape, action_size, scaler, None, augmentation
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._n_critics = n_critics
        self._target_reduction_type = target_reduction_type
        self._use_gpu = use_gpu

        # initialized in build
        self._q_func = None
        self._targ_q_func = None
        self._optim = None

    def build(self) -> None:
        # setup torch models
        self._build_network()

        # setup target network
        self._targ_q_func = copy.deepcopy(self._q_func)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._build_optim()

    def _build_network(self) -> None:
        self._q_func = create_discrete_q_function(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def _build_optim(self) -> None:
        assert self._q_func is not None
        self._optim = self._optim_factory.create(
            self._q_func.parameters(), lr=self._learning_rate
        )

    @train_api
    @torch_api(scaler_targets=["obs_t", "obs_tpn"])
    def update(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        obs_tpn: torch.Tensor,
        ter_tpn: torch.Tensor,
        n_steps: torch.Tensor,
        masks: Optional[torch.Tensor],
    ) -> np.ndarray:
        assert self._optim is not None

        self._optim.zero_grad()

        q_tpn = self.compute_target(obs_tpn)

        loss = self.compute_loss(
            obs_t, act_t, rew_tpn, q_tpn, ter_tpn, n_steps, masks
        )

        loss.backward()
        self._optim.step()

        return loss.cpu().detach().numpy()

    @augmentation_api(targets=["obs_t"])
    def compute_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        ter_tpn: torch.Tensor,
        n_steps: torch.Tensor,
        masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self._compute_loss(
            obs_t, act_t.long(), rew_tpn, q_tpn, ter_tpn, n_steps, masks
        )

    def _compute_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        ter_tpn: torch.Tensor,
        n_steps: torch.Tensor,
        masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            obs_t,
            act_t,
            rew_tpn,
            q_tpn,
            ter_tpn,
            self._gamma ** n_steps,
            use_independent_target=self._target_reduction_type == "none",
            masks=masks,
        )

    @augmentation_api(targets=["x"])
    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            max_action = self._targ_q_func(x).argmax(dim=1)
            return self._targ_q_func.compute_target(
                x, max_action, reduction=self._target_reduction_type
            )

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func(x).argmax(dim=1)

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_best_action(x)

    def update_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        hard_sync(self._targ_q_func, self._q_func)


class DoubleDQNImpl(DQNImpl):
    @augmentation_api(targets=["x"])
    def compute_target(self, x: torch.Tensor) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            action = self._predict_best_action(x)
            return self._targ_q_func.compute_target(
                x, action, reduction=self._target_reduction_type
            )
