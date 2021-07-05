import copy
from abc import abstractmethod
from typing import Optional, Sequence

import numpy as np
import torch
from torch.optim import Optimizer

from ...algos.torch.base import TorchImplBase
from ...algos.torch.utility import (
    ContinuousQFunctionMixin,
    DiscreteQFunctionMixin,
)
from ...augmentation import DrQPipeline
from ...gpu import Device
from ...models.builders import (
    create_continuous_q_function,
    create_discrete_q_function,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import (
    EnsembleContinuousQFunction,
    EnsembleDiscreteQFunction,
    EnsembleQFunction,
)
from ...preprocessing import ActionScaler, Scaler
from ...torch_utility import hard_sync, torch_api, train_api


class FQEBaseImpl(TorchImplBase):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _n_critics: int
    _use_gpu: Optional[Device]
    _q_func: Optional[EnsembleQFunction]
    _targ_q_func: Optional[EnsembleQFunction]
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
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
    ):
        super().__init__(
            observation_shape, action_size, scaler, action_scaler, DrQPipeline()
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._n_critics = n_critics
        self._use_gpu = use_gpu

        # initialized in build
        self._q_func = None
        self._targ_q_func = None
        self._optim = None

    def build(self) -> None:
        self._build_network()

        self._targ_q_func = copy.deepcopy(self._q_func)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        self._build_optim()

    @abstractmethod
    def _build_network(self) -> None:
        pass

    def _build_optim(self) -> None:
        assert self._q_func is not None
        self._optim = self._optim_factory.create(
            self._q_func.parameters(), lr=self._learning_rate
        )

    @train_api
    @torch_api(
        scaler_targets=["obs_t", "obs_tpn"],
        action_scaler_targets=["act_t", "act_tpn"],
    )
    def update(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        act_tpn: torch.Tensor,
        obs_tpn: torch.Tensor,
        ter_tpn: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> np.ndarray:
        assert self._optim is not None

        q_tpn = self.compute_target(obs_tpn, act_tpn)
        q_tpn *= 1.0 - ter_tpn
        loss = self._compute_loss(obs_t, act_t, rew_tpn, q_tpn, n_steps)

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        return loss.cpu().detach().numpy()

    def _compute_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            obs_t, act_t, rew_tpn, q_tpn, self._gamma ** n_steps
        )

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            return self._targ_q_func.compute_target(x, action)

    def update_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        hard_sync(self._targ_q_func, self._q_func)

    def save_policy(self, fname: str, as_onnx: bool) -> None:
        raise NotImplementedError


class FQEImpl(ContinuousQFunctionMixin, FQEBaseImpl):

    _q_func: Optional[EnsembleContinuousQFunction]
    _targ_q_func: Optional[EnsembleContinuousQFunction]

    def _build_network(self) -> None:
        self._q_func = create_continuous_q_function(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )


class DiscreteFQEImpl(DiscreteQFunctionMixin, FQEBaseImpl):

    _q_func: Optional[EnsembleDiscreteQFunction]
    _targ_q_func: Optional[EnsembleDiscreteQFunction]

    def _build_network(self) -> None:
        self._q_func = create_discrete_q_function(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def _compute_loss(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        rew_tpn: torch.Tensor,
        q_tpn: torch.Tensor,
        n_steps: torch.Tensor,
    ) -> torch.Tensor:
        return super()._compute_loss(
            obs_t, act_t.long(), rew_tpn, q_tpn, n_steps
        )

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return super().compute_target(x, action.long())
