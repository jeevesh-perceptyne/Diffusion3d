from typing import Dict, List, Optional
import torch
import numpy as np
import copy
import os
import cv2
from typing import Tuple
from PIL import Image
from termcolor import cprint
import open3d as o3d
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
# from diffusion_policy_3d.common.point_cloud_util import depth_to_point_cloud, fps_sampling_torch


class FrankaDataset(BaseDataset):
    def __init__(self,
                 dataset_path: str,
                 horizon: int = 16,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 seed: int = 42,
                 val_ratio: float = 0.2,
                 max_train_episodes: Optional[int] = None,
                 task_name: str = 'franka_custom',
                 use_point_cloud: bool = True,
                 num_points: int = 1024,
                 point_cloud_sampling_method: str = 'fps',
                 camera_intrinsics: Dict = None,
                 workspace_bounds: List[List[float]] = None,
                 camera_names: List[str] = None,
                 pcd_type: str = 'merged_1024'):  # New parameter to choose PCD type
        
        self.dataset_path = dataset_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_point_cloud = use_point_cloud
        self.num_points = num_points
        self.point_cloud_sampling_method = point_cloud_sampling_method
        self.pcd_type = pcd_type  # 'merged_1024', 'merged_4000', or 'wrist_pcd'
        
        
        
        # Default workspace bounds
        self.workspace_bounds = workspace_bounds or [
            [-0.1, 0.8], [-0.4, 0.45], [-0.1, 0.9]
        ]
        
        # Default camera names
        self.camera_names = camera_names or ['left_camera', 'right_camera', 'wrist_camera']
        
        # Load episodes
        self.episodes = self._load_episodes(max_train_episodes)
        
        # Split episodes into train/val
        np.random.seed(seed)
        n_episodes = len(self.episodes)
        n_val = int(n_episodes * val_ratio)
        indices = np.random.permutation(n_episodes)
        
        self.val_indices = indices[:n_val]
        self.train_indices = indices[n_val:]
        
        # Create episode index mapping
        self.episode_indices = self.train_indices
        self.is_val = False
        
        # Create sample indices
        self.sample_indices = self._create_sample_indices()
        
        # Create replay buffer for normalization
        self.replay_buffer = self._create_replay_buffer()
        
        print(f"FrankaDataset initialized:")
        print(f"  Total episodes: {n_episodes}")
        print(f"  Train episodes: {len(self.train_indices)}")
        print(f"  Val episodes: {len(self.val_indices)}")
        print(f"  Total samples: {len(self.sample_indices)}")
        print(f"  Using PCD type: {self.pcd_type}")
        print(f"  Target points per cloud: {self.num_points}")
    
    def _create_replay_buffer(self) -> Dict:
        """Create replay buffer with all training data for normalization"""
        all_actions = []
        all_states = []
        all_point_clouds = []
        
        print("Creating replay buffer for normalization...")
        
        for i, (ep_idx, start_idx) in enumerate(self.sample_indices):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(self.sample_indices)}")
                
            episode = self.episodes[ep_idx]
            
            # Extract sequences
            end_idx = start_idx + self.horizon
            states = episode['states'][start_idx:end_idx]  # (horizon, 8)
            actions = episode['actions'][start_idx:end_idx]  # (horizon, 8)
            
            # Handle padding if needed
            if len(states) < self.horizon:
                pad_size = self.horizon - len(states)
                states = np.vstack([states, np.tile(states[-1], (pad_size, 1))])
                actions = np.vstack([actions, np.tile(actions[-1], (pad_size, 1))])
            
            # Load point cloud from PCD file
            if self.use_point_cloud:
                point_cloud = self._load_point_cloud_from_pcd(
                    episode['episode_path'], 
                    start_idx
                )
            else:
                point_cloud = np.zeros((self.num_points, 3), dtype=np.float32)
            all_actions.append(actions)
            all_states.append(states[0])  # Current state only
            all_point_clouds.append(point_cloud)
        
        # Convert to numpy arrays
        all_actions = np.array(all_actions)  # (n_samples, horizon, action_dim)
        all_states = np.array(all_states)    # (n_samples, state_dim)
        all_point_clouds = np.array(all_point_clouds)  # (n_samples, num_points, 3)
        
        print(f"Replay buffer created:")
        print(f"  Actions shape: {all_actions.shape}")
        print(f"  States shape: {all_states.shape}")
        print(f"  Point clouds shape: {all_point_clouds.shape}")
        
        return {
            'action': all_actions,
            'state': all_states,
            'point_cloud': all_point_clouds
        }
    
    def _load_episodes(self, max_train_episodes: Optional[int] = None) -> List[Dict]:
        """Load all episodes from the dataset directory"""
        episodes = []
        episode_dirs = sorted([d for d in os.listdir(self.dataset_path) 
                              if d.startswith('episode_')])
        
        # Limit episodes if specified
        if max_train_episodes is not None:
            episode_dirs = episode_dirs[:max_train_episodes]
            print(f"Limited to first {max_train_episodes} episodes: {episode_dirs}")
        
        for episode_dir in episode_dirs:
            episode_path = os.path.join(self.dataset_path, episode_dir)
            
            # Load episode data
            data_path = os.path.join(episode_path, 'episode_data.npz')
            if not os.path.exists(data_path):
                print(f"Warning: No episode_data.npz found in {episode_path}")
                continue
            
            # Check if PCD files exist
            pcd_dir = os.path.join(episode_path, self.pcd_type)
            if not os.path.exists(pcd_dir):
                print(f"Warning: No {self.pcd_type} directory found in {episode_path}")
                continue
            
            # Count available PCD files
            pcd_files = [f for f in os.listdir(pcd_dir) if f.endswith('.pcd')]
            if len(pcd_files) == 0:
                print(f"Warning: No PCD files found in {pcd_dir}")
                continue
            
            data = np.load(data_path)
            
            # Extract data
            joint_states = data['joint_states'].astype(np.float32)  # (T, 7)
            gripper_states = data['gripper_states'].astype(np.float32)  # (T, 1)
            gello_joints = data['gello_joint_states'].astype(np.float32)  # (T, 7)
            gello_gripper = data['gello_gripper_percent'].astype(np.float32)  # (T, 1)
            end_effector_states = data['end_effector_states'].astype(np.float32)  # (T, 7)
            timestamps = data['timestamps'].astype(np.float64)  # (T,)

            cprint(
                f"joint_states shape: {joint_states.shape}, "
                f"gripper_states shape: {gripper_states.shape}, "
                f"gello_joints shape: {gello_joints.shape}, "
                f"gello_gripper shape: {gello_gripper.shape}, "
                f"end_effector_states shape: {end_effector_states.shape}, "
                f"timestamps shape: {timestamps.shape}"
            )
            # Ensure data shapes are consistent by reshaping if necessary
            if joint_states.ndim == 1:
                joint_states = joint_states.reshape(-1, 7)
            if gripper_states.ndim == 1:
                gripper_states = gripper_states.reshape(-1, 1)
            if gello_joints.ndim == 1:
                gello_joints = gello_joints.reshape(-1, 7)
            if gello_gripper.ndim == 1:
                gello_gripper = gello_gripper.reshape(-1, 1)
            if end_effector_states.ndim == 1:
                end_effector_states = end_effector_states.reshape(-1, 7)
            if timestamps.ndim == 1:
                timestamps = timestamps.reshape(-1)
            # Actions are joint states + gripper states (8D)
            actions = np.concatenate([gello_joints, gello_gripper], axis=1)  # (T, 8)
            
            # Observations are also joint states + gripper states (8D)
            states = np.concatenate([joint_states, gripper_states], axis=1)  # (T, 8)
            
            # Ensure we don't have more frames than PCD files
            max_frames = min(len(states), len(pcd_files))
            states = states[:max_frames]
            actions = actions[:max_frames]
            timestamps = timestamps[:max_frames]
            
            episode_data = {
                'episode_path': episode_path,
                'states': states,
                'actions': actions,
                'timestamps': timestamps,
                'episode_length': max_frames,
                'pcd_dir': pcd_dir,
                'available_pcd_files': len(pcd_files)
            }
            
            episodes.append(episode_data)
            print(f"Loaded episode {episode_dir}: {max_frames} frames, {len(pcd_files)} PCD files")
            print(f"  States shape: {states.shape} (joint + gripper)")
            print(f"  Actions shape: {actions.shape} (joint + gripper)")
        
        print(f"Total episodes loaded: {len(episodes)}")
        return episodes
    
    def _create_sample_indices(self) -> List[Tuple[int, int]]:
        """Create indices for all valid samples"""
        sample_indices = []
        
        for ep_idx in self.episode_indices:
            episode = self.episodes[ep_idx]
            episode_length = episode['episode_length']
            
            # Create samples with proper padding
            for start_idx in range(episode_length):
                end_idx = start_idx + self.horizon
                if end_idx <= episode_length:
                    sample_indices.append((ep_idx, start_idx))
        
        return sample_indices
    
    def _load_point_cloud_from_pcd(self, episode_path: str, timestep: int) -> np.ndarray:
        """Load point cloud from PCD file"""
        pcd_file = os.path.join(episode_path, self.pcd_type, f"frame_{timestep:06d}.pcd")
        
        if not os.path.exists(pcd_file):
            print(f"Warning: PCD file not found: {pcd_file}")
            return np.zeros((self.num_points, 3), dtype=np.float32)
        
        try:
            # Try to import open3d, fallback to manual parsing if not available
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(pcd_file)
                points = np.asarray(pcd.points, dtype=np.float32)
            except ImportError:
                # Fallback: simple PCD file parsing
                points = self._parse_pcd_file_manually(pcd_file)
            
            if len(points) == 0:
                print(f"Warning: Empty point cloud from {pcd_file}")
                return np.zeros((self.num_points, 3), dtype=np.float32)
            
            # Subsample or pad to desired number of points
            if len(points) > self.num_points:
                # Random sampling
                indices = np.random.choice(len(points), self.num_points, replace=False)
                points = points[indices]
            elif len(points) < self.num_points:
                # Pad with zeros or repeat points
                if len(points) > 0:
                    # Repeat points to fill
                    repeat_factor = self.num_points // len(points)
                    remainder = self.num_points % len(points)
                    points = np.vstack([
                        np.tile(points, (repeat_factor, 1)),
                        points[:remainder]
                    ])
                else:
                    points = np.zeros((self.num_points, 3), dtype=np.float32)
            
            return points
            
        except Exception as e:
            print(f"Error loading PCD file {pcd_file}: {e}")
            return np.zeros((self.num_points, 3), dtype=np.float32)
    
    def _parse_pcd_file_manually(self, pcd_file: str) -> np.ndarray:
        """Manual PCD file parsing when open3d is not available"""
        points = []
        try:
            with open(pcd_file, 'r') as f:
                data_section = False
                for line in f:
                    if line.startswith('DATA'):
                        data_section = True
                        continue
                    
                    if data_section:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            try:
                                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                                points.append([x, y, z])
                            except ValueError:
                                continue
            
            return np.array(points, dtype=np.float32)
            
        except Exception as e:
            print(f"Error parsing PCD file manually: {e}")
            return np.array([], dtype=np.float32).reshape(0, 3)
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        ep_idx, start_idx = self.sample_indices[idx]
        episode = self.episodes[ep_idx]

        end_idx = start_idx + self.horizon
        states = episode['states'][start_idx:end_idx]  # (T, 8)
        actions = episode['actions'][start_idx:end_idx]  # (T, 8)

        if len(states) < self.horizon:
            pad_size = self.horizon - len(states)
            states = np.vstack([states, np.tile(states[-1], (pad_size, 1))])
            actions = np.vstack([actions, np.tile(actions[-1], (pad_size, 1))])

        # ✅ Load sequence of point clouds
        if self.use_point_cloud:
            pcs = []
            for t in range(start_idx, start_idx + self.horizon):
                pcs.append(self._load_point_cloud_from_pcd(
                    episode['episode_path'], 
                    t
                ))
            point_cloud = np.stack(pcs, axis=0)  # [T, N, 3]
        else:
            point_cloud = np.zeros((self.horizon, self.num_points, 3), dtype=np.float32)

        return {
            'obs': {
                'point_cloud': torch.from_numpy(point_cloud).float(),  # [T, N, 3]
                'agent_pos': torch.from_numpy(states).float(),         # [T, D]
            },
            'action': torch.from_numpy(actions).float(),               # [T, D]
        }

    
    def get_validation_dataset(self):
        """Return validation dataset"""
        val_dataset = FrankaDataset.__new__(FrankaDataset)
        val_dataset.__dict__.update(self.__dict__)
        val_dataset.episode_indices = self.val_indices
        val_dataset.is_val = True
        val_dataset.sample_indices = val_dataset._create_sample_indices()
        
        # Create replay buffer for validation dataset
        val_dataset.replay_buffer = val_dataset._create_replay_buffer()
        
        return val_dataset
    
    def get_normalizer(self, mode='limits', **kwargs):
        """Return normalizer for the dataset using LinearNormalizer"""
        from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
        
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        return normalizer