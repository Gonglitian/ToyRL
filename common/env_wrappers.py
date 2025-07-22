import numpy as np
import cv2
import gymnasium as gym
import ale_py  # Import to register ALE environments
from gymnasium import spaces
from collections import deque
from typing import Optional, Union


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """
    
    def __init__(self, env: gym.Env, noop_max: int = 30):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
            noop_max: Maximum number of no-ops to sample
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    
    def reset(self, **kwargs):
        """Reset environment with random no-ops."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, _ = self.env.reset(**kwargs)
        return obs, {}


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame and max over the last two frames.
    """
    
    def __init__(self, env: gym.Env, skip: int = 4):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
            skip: Number of frames to skip
        """
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
    
    def step(self, action):
        """Run action for skip frames and return max over last two."""
        total_reward = 0.0
        terminated = truncated = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        
        # Take maximum over last two frames
        max_frame = self._obs_buffer.max(axis=0)
        
        return max_frame, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment."""
        return self.env.reset(**kwargs)


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """
    
    def __init__(self, env: gym.Env):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
    
    def step(self, action):
        """Step and reset if life lost."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        
        # Check current lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # Life lost but game not over
            terminated = True
        self.lives = lives
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset if game is really done."""
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # No-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.
    """
    
    def __init__(self, env: gym.Env):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
        """
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self, **kwargs):
        """Reset and fire."""
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            obs, _ = self.env.reset(**kwargs)
        return obs, {}


class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to 84x84 as done in the Nature paper and later work.
    """
    
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, grayscale: bool = True):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
            width: Target width
            height: Target height
            grayscale: Whether to convert to grayscale
        """
        super(WarpFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        
        if self.grayscale:
            num_colors = 1
        else:
            num_colors = 3
        
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, num_colors),
            dtype=np.uint8,
        )
        self.observation_space = new_space
    
    def observation(self, frame):
        """Warp frame to target size."""
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Normalize pixel values to [0, 1].
    """
    
    def __init__(self, env: gym.Env):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
        """
        super(ScaledFloatFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=env.observation_space.shape,
            dtype=np.float32
        )
    
    def observation(self, observation):
        """Scale observation to [0, 1]."""
        return np.array(observation).astype(np.float32) / 255.0


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip rewards to {-1, 0, 1} by their sign.
    """
    
    def __init__(self, env: gym.Env):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
        """
        super(ClipRewardEnv, self).__init__(env)
    
    def reward(self, reward):
        """Clip reward by sign."""
        return np.sign(reward)


class FrameStack(gym.Wrapper):
    """
    Stack k last frames.
    """
    
    def __init__(self, env: gym.Env, k: int):
        """
        Initialize wrapper.
        
        Args:
            env: Environment to wrap
            k: Number of frames to stack
        """
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        
        low = np.repeat(self.observation_space.low[np.newaxis, ...], k, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], k, axis=0)
        
        # If the original space has a single channel dimension, remove it
        if low.ndim == 4 and low.shape[-1] == 1:
            low = low.squeeze(-1)
            high = high.squeeze(-1)
        
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=self.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        """Reset and clear frame stack."""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        """Step and add frame to stack."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        """Get stacked observation."""
        assert len(self.frames) == self.k
        stacked = np.array(self.frames)
        # Convert from (stack, height, width, channels) to (stack*channels, height, width)
        # For grayscale images with shape (4, 84, 84, 1), we want (4, 84, 84)
        if stacked.ndim == 4 and stacked.shape[-1] == 1:
            stacked = stacked.squeeze(-1)  # Remove last dimension
        return stacked


def make_atari_env(env_id: str, stack_frames: int = 4, episodic_life: bool = True,
                   clip_rewards: bool = True, frame_skip: int = 4, screen_size: int = 84):
    """
    Create a wrapped Atari environment.
    
    Args:
        env_id: Environment ID
        stack_frames: Number of frames to stack
        episodic_life: Whether to use episodic life
        clip_rewards: Whether to clip rewards
        frame_skip: Number of frames to skip
        screen_size: Screen size for warping
        
    Returns:
        Wrapped environment
    """
    env = gym.make(env_id, render_mode=None)
    
    # Standard Atari preprocessing
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    
    if episodic_life:
        env = EpisodicLifeEnv(env)
    
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    
    env = WarpFrame(env, width=screen_size, height=screen_size)
    
    if clip_rewards:
        env = ClipRewardEnv(env)
    
    env = ScaledFloatFrame(env)
    env = FrameStack(env, stack_frames)
    
    return env