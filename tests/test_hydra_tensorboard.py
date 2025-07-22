"""
Unit tests for Hydra configuration and TensorBoard logging integration.
"""

import os
import tempfile
import shutil
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

import hydra
from omegaconf import DictConfig, OmegaConf

from common.logger import Logger
from common.env_wrappers import make_atari_env


class TestLogger:
    """Test the enhanced Logger class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = Logger(self.temp_dir)
        
    def tearDown(self):
        """Clean up test environment."""
        self.logger.close()
        shutil.rmtree(self.temp_dir)
        
    def test_logger_initialization(self):
        """Test logger initialization."""
        self.setUp()
        assert os.path.exists(self.temp_dir)
        assert self.logger.global_step == 0
        assert self.logger.episode_count == 0
        self.tearDown()
        
    def test_log_scalar(self):
        """Test scalar logging."""
        self.setUp()
        self.logger.log_scalar("test_metric", 1.5, 10)
        assert "test_metric" in self.logger.metrics
        assert self.logger.metrics["test_metric"][0] == 1.5
        self.tearDown()
        
    def test_log_histogram(self):
        """Test histogram logging."""
        self.setUp()
        values = torch.randn(100)
        self.logger.log_histogram("test_hist", values, 10)
        # Should not raise any exceptions
        self.tearDown()
        
    def test_log_episode_reward(self):
        """Test episode reward logging."""
        self.setUp()
        self.logger.log_episode_reward(10.5, 5)
        assert "Episode/Reward" in self.logger.metrics
        assert self.logger.metrics["Episode/Reward"][0] == 10.5
        self.tearDown()
        
    def test_step_counting(self):
        """Test step counting functionality."""
        self.setUp()
        self.logger.step()
        assert self.logger.global_step == 1
        self.logger.episode_step()
        assert self.logger.episode_count == 1
        self.tearDown()


class TestHydraConfig:
    """Test Hydra configuration system."""
    
    def test_config_loading(self):
        """Test that configs can be loaded properly."""
        # Test that config files exist
        assert os.path.exists("configs/trainer.yaml")
        assert os.path.exists("configs/algorithm/dqn.yaml")
        assert os.path.exists("configs/algorithm/ppo.yaml")
        assert os.path.exists("configs/env/atari.yaml")
        
    def test_config_composition(self):
        """Test config composition works."""
        with hydra.initialize(config_path="../configs", version_base="1.3"):
            cfg = hydra.compose(config_name="trainer")
            
            # Check that basic config is loaded
            assert "seed" in cfg
            assert "episodes" in cfg
            assert "log_dir" in cfg
            
            # Check that algorithm config is included
            assert "algorithm_name" in cfg
            
            # Check that environment config is included
            assert "env_name" in cfg


class TestTrainingIntegration:
    """Test training integration with TensorBoard."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
        
    @patch('train.make_atari_env')
    @patch('train.create_agent')
    def test_short_training_run(self, mock_create_agent, mock_make_env):
        """Test a short training run to verify TensorBoard logging."""
        self.setUp()
        
        # Mock environment
        mock_env = MagicMock()
        mock_env.observation_space.shape = (4, 84, 84)
        mock_env.action_space.n = 4
        mock_env.reset.return_value = (np.zeros((4, 84, 84)), {})
        mock_env.step.return_value = (np.zeros((4, 84, 84)), 1.0, True, False, {})
        mock_make_env.return_value = mock_env
        
        # Mock agent
        mock_agent = MagicMock()
        mock_agent.select_action.return_value = 0
        mock_agent.store_transition.return_value = None
        mock_agent.train_step.return_value = 0.5
        mock_agent.save_model.return_value = None
        mock_create_agent.return_value = mock_agent
        
        # Create test config
        config = DictConfig({
            'algorithm_name': 'dqn',
            'env_name': 'PongNoFrameskip-v4',
            'episodes': 2,  # Very short run
            'max_steps': 10,
            'eval_freq': 1,
            'eval_episodes': 1,
            'save_freq': 10,
            'seed': 42,
            'log_dir': self.temp_dir,
            'save_dir': os.path.join(self.temp_dir, 'models'),
            'tensorboard': {
                'log_interval': 1,
                'log_histograms': True,
                'log_model_graph': True
            }
        })
        
        # Import here to avoid circular imports
        from train import train_with_config
        
        # Run training
        train_with_config(config)
        
        # Check that TensorBoard event files were created
        event_files = []
        for root, dirs, files in os.walk(self.temp_dir):
            for file in files:
                if file.startswith('events.out.tfevents'):
                    event_files.append(file)
        
        assert len(event_files) > 0, "No TensorBoard event files were created"
        self.tearDown()


def test_backward_compatibility():
    """Test that argparse mode still works."""
    # This test ensures the old argparse interface still functions
    from train import parse_args
    
    # Mock sys.argv for testing
    import sys
    original_argv = sys.argv.copy()
    
    try:
        sys.argv = ['train.py', '--use_argparse', '--algorithm', 'dqn', '--episodes', '1']
        args = parse_args()
        assert args.use_argparse == True
        assert args.algorithm == 'dqn'
        assert args.episodes == 1
    finally:
        sys.argv = original_argv


def test_config_override():
    """Test that Hydra config overrides work."""
    with hydra.initialize(config_path="../configs", version_base="1.3"):
        # Test basic override
        cfg = hydra.compose(config_name="trainer", overrides=["episodes=500"])
        assert cfg.episodes == 500
        
        # Test algorithm override
        cfg = hydra.compose(config_name="trainer", overrides=["algorithm=ppo"])
        assert cfg.algorithm_name == "ppo"
        
        # Test environment override
        cfg = hydra.compose(config_name="trainer", overrides=["env_name=BreakoutNoFrameskip-v4"])
        assert cfg.env_name == "BreakoutNoFrameskip-v4"


if __name__ == "__main__":
    # Run the tests
    test_logger = TestLogger()
    test_logger.test_logger_initialization()
    test_logger.test_log_scalar()
    test_logger.test_log_histogram()
    test_logger.test_log_episode_reward()
    test_logger.test_step_counting()
    
    test_hydra = TestHydraConfig()
    test_hydra.test_config_loading()
    test_hydra.test_config_composition()
    
    test_integration = TestTrainingIntegration()
    test_integration.test_short_training_run()
    
    test_backward_compatibility()
    test_config_override()
    
    print("All tests passed!")