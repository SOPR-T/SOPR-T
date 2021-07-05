import gym
import pytest
from gym.wrappers import AtariPreprocessing

from d3rlpy.algos import DQN
from d3rlpy.envs.wrappers import Atari, ChannelFirst


def test_channel_first():
    env = gym.make("Breakout-v0")

    width, height, channel = env.observation_space.shape

    wrapper = ChannelFirst(env)

    # check reset
    observation = wrapper.reset()
    assert observation.shape == (channel, width, height)

    # check step
    observation, _, _, _ = wrapper.step(wrapper.action_space.sample())
    assert observation.shape == (channel, width, height)

    # check with algorithm
    dqn = DQN()
    dqn.build_with_env(wrapper)
    dqn.predict([observation])


def test_channel_first_with_2_dim_obs():
    env = AtariPreprocessing(gym.make("BreakoutNoFrameskip-v4"))

    width, height = env.observation_space.shape

    wrapper = ChannelFirst(env)

    # check reset
    observation = wrapper.reset()
    assert observation.shape == (1, width, height)

    # check step
    observation, _, _, _ = wrapper.step(wrapper.action_space.sample())
    assert observation.shape == (1, width, height)

    # check with algorithm
    dqn = DQN()
    dqn.build_with_env(wrapper)
    dqn.predict([observation])


@pytest.mark.parametrize("is_eval", [100])
def test_atari(is_eval):
    env = Atari(gym.make("BreakoutNoFrameskip-v4"), is_eval)

    assert env.observation_space.shape == (1, 84, 84)

    # check reset
    observation = env.reset()
    assert observation.shape == (1, 84, 84)

    # check step
    observation, reward, done, info = env.step(env.action_space.sample())
    assert observation.shape == (1, 84, 84)
