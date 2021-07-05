# pylint: disable=unused-import

import os
import pickle
import re
import urllib.request as request
from typing import Tuple

import gym
import numpy as np

from .dataset import MDPDataset
from .envs import ChannelFirst

DATA_DIRECTORY = "d3rlpy_data"
CARTPOLE_URL = "https://www.dropbox.com/s/l1sdnq3zvoot2um/cartpole.h5?dl=1"
PENDULUM_URL = "https://www.dropbox.com/s/vsiz9pwvshj7sly/pendulum.h5?dl=1"


def get_cartpole(
    create_mask: bool = False, mask_size: int = 1
) -> Tuple[MDPDataset, gym.Env]:
    """Returns cartpole dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/cartpole.h5`` if
    it does not exist.

    Args:
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    data_path = os.path.join(DATA_DIRECTORY, "cartpole.h5")

    # download dataset
    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print("Donwloading cartpole.pkl into %s..." % data_path)
        request.urlretrieve(CARTPOLE_URL, data_path)

    # load dataset
    dataset = MDPDataset.load(
        data_path, create_mask=create_mask, mask_size=mask_size
    )

    # environment
    env = gym.make("CartPole-v0")

    return dataset, env


def get_pendulum(
    create_mask: bool = False, mask_size: int = 1
) -> Tuple[MDPDataset, gym.Env]:
    """Returns pendulum dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/pendulum.h5`` if
    it does not exist.

    Args:
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    data_path = os.path.join(DATA_DIRECTORY, "pendulum.h5")

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print("Donwloading pendulum.pkl into %s..." % data_path)
        request.urlretrieve(PENDULUM_URL, data_path)

    # load dataset
    dataset = MDPDataset.load(
        data_path, create_mask=create_mask, mask_size=mask_size
    )

    # environment
    env = gym.make("Pendulum-v0")

    return dataset, env


def get_pybullet(
    env_name: str, create_mask: bool = False, mask_size: int = 1
) -> Tuple[MDPDataset, gym.Env]:
    """Returns pybullet dataset and envrironment.

    The dataset is provided through d4rl-pybullet. See more details including
    available dataset from its GitHub page.

    .. code-block:: python

        from d3rlpy.datasets import get_pybullet

        dataset, env = get_pybullet('hopper-bullet-mixed-v0')

    References:
        * https://github.com/takuseno/d4rl-pybullet

    Args:
        env_name: environment id of d4rl-pybullet dataset.
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    try:
        import d4rl_pybullet  # type: ignore

        env = gym.make(env_name)
        dataset = MDPDataset(
            create_mask=create_mask, mask_size=mask_size, **env.get_dataset()
        )
        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-pybullet is not installed.\n"
            "pip install git+https://github.com/takuseno/d4rl-pybullet"
        ) from e


def get_atari(
    env_name: str, create_mask: bool = False, mask_size: int = 1
) -> Tuple[MDPDataset, gym.Env]:
    """Returns atari dataset and envrironment.

    The dataset is provided through d4rl-atari. See more details including
    available dataset from its GitHub page.

    .. code-block:: python

        from d3rlpy.datasets import get_atari

        dataset, env = get_atari('breakout-mixed-v0')

    References:
        * https://github.com/takuseno/d4rl-atari

    Args:
        env_name: environment id of d4rl-atari dataset.
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    try:
        import d4rl_atari  # type: ignore

        env = ChannelFirst(gym.make(env_name))
        dataset = MDPDataset(
            discrete_action=True,
            create_mask=create_mask,
            mask_size=mask_size,
            **env.get_dataset(),
        )
        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n"
            "pip install git+https://github.com/takuseno/d4rl-atari"
        ) from e


def get_d4rl(
    env_name: str, create_mask: bool = False, mask_size: int = 1
) -> Tuple[MDPDataset, gym.Env]:
    """Returns d4rl dataset and envrironment.

    The dataset is provided through d4rl.

    .. code-block:: python

        from d3rlpy.datasets import get_d4rl

        dataset, env = get_d4rl('hopper-medium-v0')

    References:
        * `Fu et al., D4RL: Datasets for Deep Data-Driven Reinforcement
          Learning. <https://arxiv.org/abs/2004.07219>`_
        * https://github.com/rail-berkeley/d4rl

    Args:
        env_name: environment id of d4rl dataset.
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    try:
        import d4rl  # type: ignore

        env = gym.make(env_name)
        dataset = env.get_dataset()

        observations = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        terminals = np.logical_and(
            dataset["terminals"], np.logical_not(dataset["timeouts"])
        )
        episode_terminals = np.logical_or(
            dataset["terminals"], dataset["timeouts"]
        )

        mdp_dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            episode_terminals=episode_terminals,
            create_mask=create_mask,
            mask_size=mask_size,
        )

        return mdp_dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl is not installed.\n"
            "pip install git+https://github.com/rail-berkeley/d4rl"
        ) from e


ATARI_GAMES = [
    "adventure",
    "air-raid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bank-heist",
    "battle-zone",
    "beam-rider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "centipede",
    "chopper-command",
    "crazy-climber",
    "defender",
    "demon-attack",
    "double-dunk",
    "elevator-action",
    "enduro",
    "fishing-derby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "ice-hockey",
    "jamesbond",
    "journey-escape",
    "kangaroo",
    "krull",
    "kung-fu-master",
    "montezuma-revenge",
    "ms-pacman",
    "name-this-game",
    "phoenix",
    "pitfall",
    "pong",
    "pooyan",
    "private-eye",
    "qbert",
    "riverraid",
    "road-runner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "space-invaders",
    "star-gunner",
    "tennis",
    "time-pilot",
    "tutankham",
    "up-n-down",
    "venture",
    "video-pinball",
    "wizard-of-wor",
    "yars-revenge",
    "zaxxon",
]


def get_dataset(
    env_name: str, create_mask: bool = False, mask_size: int = 1
) -> Tuple[MDPDataset, gym.Env]:
    """Returns dataset and envrironment by guessing from name.

    This function returns dataset by matching name with the following datasets.

    - cartpole
    - pendulum
    - d4rl-pybullet
    - d4rl-atari
    - d4rl

    .. code-block:: python

       import d3rlpy

       # cartpole dataset
       dataset, env = d3rlpy.datasets.get_dataset('cartpole')

       # pendulum dataset
       dataset, env = d3rlpy.datasets.get_dataset('pendulum')

       # d4rl-pybullet dataset
       dataset, env = d3rlpy.datasets.get_dataset('hopper-bullet-mixed-v0')

       # d4rl-atari dataset
       dataset, env = d3rlpy.datasets.get_dataset('breakout-mixed-v0')

       # d4rl dataset
       dataset, env = d3rlpy.datasets.get_dataset('hopper-medium-v0')

    Args:
        env_name: environment id of the dataset.
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.

    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.

    """
    if env_name == "cartpole":
        return get_cartpole(create_mask, mask_size)
    elif env_name == "pendulum":
        return get_pendulum(create_mask, mask_size)
    elif re.match(r"^bullet-.+$", env_name):
        return get_d4rl(env_name, create_mask, mask_size)
    elif re.match(r"^.+-bullet-.+$", env_name):
        return get_pybullet(env_name, create_mask, mask_size)
    elif re.match(r"hopper|halfcheetah|walker|ant", env_name):
        return get_d4rl(env_name, create_mask, mask_size)
    elif re.match(re.compile("|".join(ATARI_GAMES)), env_name):
        return get_atari(env_name, create_mask, mask_size)
    raise ValueError(f"Unrecognized env_name: {env_name}.")
