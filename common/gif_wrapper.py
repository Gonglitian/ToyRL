"""
GIF Recording Wrapper for Environment Visualization

This wrapper allows recording environment episodes as GIF files for visualization.
"""

import os
import numpy as np
import gymnasium as gym
from typing import Optional, List, Union
from PIL import Image
import cv2


class GifRecorderWrapper(gym.Wrapper):
    """
    Wrapper that records environment episodes as GIF files.
    
    This wrapper captures frames during environment execution and saves them
    as animated GIF files. Useful for visualizing agent behavior and creating
    demonstrations.
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        save_dir: str = "gifs",
        episode_trigger: Optional[callable] = None,
        name_prefix: str = "episode",
        fps: int = 30,
        resize: Optional[tuple] = None,
        quality: int = 95
    ):
        """
        Initialize GIF recorder wrapper.
        
        Args:
            env: Environment to wrap
            save_dir: Directory to save GIF files
            episode_trigger: Function that takes episode number and returns bool
                           indicating whether to record this episode. If None,
                           records all episodes.
            name_prefix: Prefix for GIF filenames
            fps: Frames per second for the GIF
            resize: Tuple (width, height) to resize frames. If None, keeps original size
            quality: GIF quality (0-100), higher is better quality but larger file
        """
        super().__init__(env)
        self.save_dir = save_dir
        self.episode_trigger = episode_trigger
        self.name_prefix = name_prefix
        self.fps = fps
        self.resize = resize
        self.quality = quality
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Recording state
        self.frames: List[np.ndarray] = []
        self.episode_count = 0
        self.is_recording = False
        self.current_episode_frames = []
        
    def reset(self, **kwargs):
        """Reset environment and start recording if triggered."""
        obs, info = self.env.reset(**kwargs)
        
        # Save previous episode if we were recording
        if self.is_recording and self.current_episode_frames:
            self._save_gif()
        
        # Check if we should record this episode
        self.episode_count += 1
        self.is_recording = (
            self.episode_trigger is None or 
            self.episode_trigger(self.episode_count)
        )
        
        # Start new recording
        self.current_episode_frames = []
        if self.is_recording:
            frame = self._get_frame()
            if frame is not None:
                self.current_episode_frames.append(frame)
        
        return obs, info
    
    def step(self, action):
        """Step environment and record frame if recording."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if self.is_recording:
            frame = self._get_frame()
            if frame is not None:
                self.current_episode_frames.append(frame)
        
        # Save GIF if episode ended
        if (terminated or truncated) and self.is_recording and self.current_episode_frames:
            self._save_gif()
            self.is_recording = False
        
        return obs, reward, terminated, truncated, info
    
    def _get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from environment."""
        try:
            # Try to render the environment
            frame = self.env.render()
            if frame is None:
                return None
                
            # Convert to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Already RGB
                rgb_frame = frame
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                # RGBA to RGB
                rgb_frame = frame[:, :, :3]
            else:
                # Grayscale to RGB
                rgb_frame = np.stack([frame] * 3, axis=-1)
            
            # Resize if specified
            if self.resize is not None:
                rgb_frame = cv2.resize(rgb_frame, self.resize)
            
            return rgb_frame.astype(np.uint8)
            
        except Exception as e:
            print(f"Warning: Could not capture frame: {e}")
            return None
    
    def _save_gif(self):
        """Save recorded frames as GIF."""
        if not self.current_episode_frames:
            return
        
        try:
            # Convert frames to PIL Images
            images = [Image.fromarray(frame) for frame in self.current_episode_frames]
            
            # Calculate duration per frame in milliseconds
            duration = int(1000 / self.fps)
            
            # Generate filename
            filename = f"{self.name_prefix}_{self.episode_count:04d}.gif"
            filepath = os.path.join(self.save_dir, filename)
            
            # Save as GIF
            images[0].save(
                filepath,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,  # Loop forever
                optimize=True,
                quality=self.quality
            )
            
            print(f"Saved GIF: {filepath} ({len(images)} frames)")
            
        except Exception as e:
            print(f"Error saving GIF: {e}")
    
    def close(self):
        """Close wrapper and save any remaining frames."""
        if self.is_recording and self.current_episode_frames:
            self._save_gif()
        super().close()


def record_every_n_episodes(n: int):
    """Helper function to create episode trigger for recording every N episodes."""
    def trigger(episode_num: int) -> bool:
        return episode_num % n == 0
    return trigger


def record_specific_episodes(episodes: List[int]):
    """Helper function to create episode trigger for recording specific episodes."""
    def trigger(episode_num: int) -> bool:
        return episode_num in episodes
    return trigger


def record_first_and_last(total_episodes: int, first_n: int = 5, last_n: int = 5):
    """Helper function to record first N and last N episodes."""
    def trigger(episode_num: int) -> bool:
        return (episode_num <= first_n or 
                episode_num > total_episodes - last_n)
    return trigger