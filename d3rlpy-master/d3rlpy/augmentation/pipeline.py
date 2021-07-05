from typing import Any, Callable, Dict, List, Optional

import torch

from ..decorators import pretty_repr
from .base import Augmentation


@pretty_repr
class AugmentationPipeline:
    _augmentations: List[Augmentation]

    def __init__(self, augmentations: List[Augmentation]):
        self._augmentations = augmentations

    def append(self, augmentation: Augmentation) -> None:
        """Append augmentation to pipeline.

        Args:
            augmentation: augmentation.

        """
        self._augmentations.append(augmentation)

    def get_augmentation_types(self) -> List[str]:
        """Returns augmentation types.

        Returns:
            list of augmentation types.

        """
        return [aug.get_type() for aug in self._augmentations]

    def get_augmentation_params(self) -> List[Dict[str, Any]]:
        """Returns augmentation parameters.

        Args:
            deep: flag to deeply copy objects.

        Returns:
            list of augmentation parameters.

        """
        return [aug.get_params() for aug in self._augmentations]

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns pipeline parameters.

        Returns:
            piple parameters.

        """
        raise NotImplementedError

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Returns observation processed by all augmentations.

        Args:
            x: observation tensor.

        Returns:
            processed observation tensor.

        """
        if not self._augmentations:
            return x

        for augmentation in self._augmentations:
            x = augmentation.transform(x)

        return x

    def process(
        self,
        func: Callable[..., torch.Tensor],
        inputs: Dict[str, torch.Tensor],
        targets: List[str],
    ) -> torch.Tensor:
        """Runs a given function while augmenting inputs.

        Args:
            func: function to compute.
            inputs: inputs to the func.
            target: list of argument names to augment.

        Returns:
            the computation result.

        """
        raise NotImplementedError

    @property
    def augmentations(self) -> List[Augmentation]:
        return self._augmentations


class DrQPipeline(AugmentationPipeline):
    """Data-reguralized Q augmentation pipeline.

    References:
        * `Kostrikov et al., Image Augmentation Is All You Need: Regularizing
          Deep Reinforcement Learning from Pixels.
          <https://arxiv.org/abs/2004.13649>`_

    Args:
        augmentations (list(d3rlpy.augmentation.base.Augmentation or str)):
            list of augmentations or augmentation types.
        n_mean (int): the number of computations to average

    """

    _n_mean: int

    def __init__(
        self,
        augmentations: Optional[List[Augmentation]] = None,
        n_mean: int = 1,
    ):
        if augmentations is None:
            augmentations = []
        super().__init__(augmentations)
        self._n_mean = n_mean

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {"n_mean": self._n_mean}

    def process(
        self,
        func: Callable[..., torch.Tensor],
        inputs: Dict[str, torch.Tensor],
        targets: List[str],
    ) -> torch.Tensor:
        ret = 0.0
        for _ in range(self._n_mean):
            kwargs = dict(inputs)
            for target in targets:
                kwargs[target] = self.transform(kwargs[target])
            ret += func(**kwargs)
        return ret / self._n_mean
