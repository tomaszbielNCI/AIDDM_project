"""Sanity checks on the Gymnasium environment contract.

Tests that the PricingEnv follows the Gymnasium API contract.
"""

import logging

import numpy as np
import pytest

# Try to import gymnasium and the environment
try:
    import gymnasium as gym
    from archive import PricingEnv
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

logger = logging.getLogger(__name__)


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
def test_env_initialization():
    """Test that the environment can be initialized."""
    # Create dummy demand data
    demand_data = np.random.rand(100) * 10
    feature_data = np.random.rand(100, 5)

    env = PricingEnv(demand_data, feature_data, seed=42)

    assert env is not None
    assert hasattr(env, "action_space")
    assert hasattr(env, "observation_space")


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
def test_action_space():
    """Test that the action space is correctly defined."""
    demand_data = np.random.rand(100) * 10
    feature_data = np.random.rand(100, 5)

    env = PricingEnv(demand_data, feature_data, seed=42)

    # Check action space is Discrete
    assert isinstance(env.action_space, gym.spaces.Discrete)

    # Check action space has correct number of actions
    assert env.action_space.n > 0


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
def test_observation_space():
    """Test that the observation space is correctly defined."""
    demand_data = np.random.rand(100) * 10
    feature_data = np.random.rand(100, 5)

    env = PricingEnv(demand_data, feature_data, seed=42)

    # Check observation space is Box
    assert isinstance(env.observation_space, gym.spaces.Box)

    # Check observation space shape
    assert env.observation_space.shape is not None
    assert len(env.observation_space.shape) > 0


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
def test_reset():
    """Test that reset returns valid observation and info."""
    demand_data = np.random.rand(100) * 10
    feature_data = np.random.rand(100, 5)

    env = PricingEnv(demand_data, feature_data, seed=42)

    obs, info = env.reset(seed=42)

    # Check observation is in observation space
    assert env.observation_space.contains(obs)

    # Check info is a dict
    assert isinstance(info, dict)


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
def test_step():
    """Test that step returns valid outputs."""
    demand_data = np.random.rand(100) * 10
    feature_data = np.random.rand(100, 5)

    env = PricingEnv(demand_data, feature_data, seed=42)
    env.reset(seed=42)

    # Take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Check observation is in observation space
    assert env.observation_space.contains(obs)

    # Check reward is a float
    assert isinstance(reward, (float, np.floating))

    # Check terminated and truncated are booleans
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    # Check info is a dict
    assert isinstance(info, dict)


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
def test_episode_termination():
    """Test that episodes terminate correctly."""
    demand_data = np.random.rand(100) * 10
    feature_data = np.random.rand(100, 5)

    env = PricingEnv(demand_data, feature_data, seed=42)
    env.reset(seed=42)

    terminated = False
    truncated = False
    step_count = 0
    max_steps = 200

    while not terminated and not truncated and step_count < max_steps:
        action = env.action_space.sample()
        _, terminated, truncated, _, _ = env.step(action)
        step_count += 1

    # Episode should terminate within reasonable steps
    assert step_count < max_steps


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not installed")
def test_deterministic_reset():
    """Test that reset with same seed produces same initial state."""
    demand_data = np.random.rand(100) * 10
    feature_data = np.random.rand(100, 5)

    env = PricingEnv(demand_data, feature_data)

    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)

    # Observations should be identical with same seed
    np.testing.assert_array_equal(obs1, obs2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v"])
