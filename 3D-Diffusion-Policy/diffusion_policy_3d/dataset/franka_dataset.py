from typing import Dict, List, Optional
import torch
import numpy as np
import copy
import os
import cv2
import open3d as o3d
from PIL import Image
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import pytorch3d.ops as torch3d_ops


class FrankaDataset(BaseDataset):
    def __init__(self,
                 dataset_path: str,
                 horizon: int = 1,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 seed: int = 42,
                 val_ratio: float = 0.0,
                 max_train_episodes: Optional[int] = None,
                 task_name: Optional[str] = None,
                 use_point_cloud: bool = True,
                 num_points: int = 512,
                 camera_intrinsics: Optional[Dict] = None,
                 point_cloud_sampling_method: str = 'fps',
                 workspace_bounds: Optional[List] = None,
                 ):
        super().__init__()
        self.dataset_path = dataset_path
        self.task_name = task_name
        self.use_point_cloud = use_point_cloud
        self.num_points = num_points
        self.point_cloud_sampling_method = point_cloud_sampling_method
        self.workspace_bounds = workspace_bounds
        
        # Default camera intrinsics if not provided
        if camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5,
                'width': 640, 'height': 480
            }
        else:
            self.camera_intrinsics = camera_intrinsics

        # Load episode data
        self.episodes_data = self._load_episodes()
        
        # Create episode masks for train/val split
        n_episodes = len(self.episodes_data)
        val_mask = get_val_mask(
            n_episodes=n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # Create replay buffer-like structure
        self.replay_buffer = self._create_replay_buffer()
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _load_episodes(self) -> List[Dict]:
        """Load all episode data from the dataset path"""
        episodes = []
        episode_dirs = [d for d in os.listdir(self.dataset_path) 
                       if os.path.isdir(os.path.join(self.dataset_path, d)) and d.startswith('episode_')]
        
        episode_dirs.sort()  # Ensure consistent ordering
        
        for episode_dir in episode_dirs:
            episode_path = os.path.join(self.dataset_path, episode_dir)
            episode_data_path = os.path.join(episode_path, 'episode_data.npz')
            
            if os.path.exists(episode_data_path):
                # Load episode data
                episode_data = np.load(episode_data_path, allow_pickle=True)
                
                # Load depth images
                left_depth_dir = os.path.join(episode_path, 'left_depth_images')
                right_depth_dir = os.path.join(episode_path, 'right_depth_images')
                wrist_depth_dir = os.path.join(episode_path, 'wrist_depth_images')
                
                episode_info = {
                    'episode_path': episode_path,
                    'episode_data': episode_data,
                    'left_depth_dir': left_depth_dir if os.path.exists(left_depth_dir) else None,
                    'right_depth_dir': right_depth_dir if os.path.exists(right_depth_dir) else None,
                    'wrist_depth_dir': wrist_depth_dir if os.path.exists(wrist_depth_dir) else None,
                }
                episodes.append(episode_info)
        
        return episodes

    def _create_replay_buffer(self) -> ReplayBuffer:
        """Create a replay buffer from the loaded episodes"""
        all_states = []
        all_actions = []
        all_point_clouds = []
        all_images = []
        episode_ends = []
        
        total_steps = 0
        
        for episode_info in self.episodes_data:
            episode_data = episode_info['episode_data']
            
            # Extract data from episode
            states = episode_data['states'] if 'states' in episode_data else episode_data['agent_pos']
            actions = episode_data['actions'] if 'actions' in episode_data else episode_data['action']
            
            episode_length = len(states)
            
            # Process each timestep
            for step_idx in range(episode_length):
                # Get state and action
                state = states[step_idx]
                action = actions[step_idx]
                
                # Generate point cloud from depth images
                if self.use_point_cloud:
                    point_cloud = self._generate_point_cloud_from_depth(episode_info, step_idx)
                else:
                    point_cloud = np.zeros((self.num_points, 3))  # Dummy point cloud
                
                # Create dummy image for compatibility
                image = np.zeros((84, 84, 3), dtype=np.uint8)
                
                all_states.append(state)
                all_actions.append(action)
                all_point_clouds.append(point_cloud)
                all_images.append(image)
                total_steps += 1
            
            episode_ends.append(total_steps)
        
        # Convert to numpy arrays
        states_array = np.array(all_states)
        actions_array = np.array(all_actions)
        point_clouds_array = np.array(all_point_clouds)
        images_array = np.array(all_images)
        
        # Create replay buffer
        data = {
            'state': states_array,
            'action': actions_array,
            'point_cloud': point_clouds_array,
            'img': images_array,
        }
        
        replay_buffer = ReplayBuffer(data=data, episode_ends=episode_ends)
        return replay_buffer

    def _generate_point_cloud_from_depth(self, episode_info: Dict, step_idx: int) -> np.ndarray:
        """Generate point cloud from depth images for a specific timestep"""
        point_clouds = []
        
        # Process each camera's depth image
        for depth_dir_key in ['left_depth_dir', 'right_depth_dir', 'wrist_depth_dir']:
            depth_dir = episode_info[depth_dir_key]
            if depth_dir is not None and os.path.exists(depth_dir):
                depth_img_path = os.path.join(depth_dir, f'frame_{step_idx:06d}.png')
                
                if os.path.exists(depth_img_path):
                    # Load depth image
                    depth_img = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
                    if depth_img is not None:
                        # Convert depth image to point cloud
                        pc = self._depth_to_point_cloud(depth_img)
                        if pc is not None and len(pc) > 0:
                            point_clouds.append(pc)
        
        # Combine point clouds from all cameras
        if len(point_clouds) > 0:
            combined_pc = np.vstack(point_clouds)
            
            # Apply workspace bounds if specified
            if self.workspace_bounds is not None:
                combined_pc = self._filter_by_workspace(combined_pc)
            
            # Sample to fixed number of points
            combined_pc = self._sample_point_cloud(combined_pc, self.num_points)
            
            # Ensure we have the right number of points
            if len(combined_pc) < self.num_points:
                # Pad with zeros
                padding = np.zeros((self.num_points - len(combined_pc), combined_pc.shape[1]))
                combined_pc = np.vstack([combined_pc, padding])
            
            return combined_pc[:self.num_points]
        else:
            # Return dummy point cloud if no depth images available
            return np.zeros((self.num_points, 3))

    def _depth_to_point_cloud(self, depth_img: np.ndarray) -> Optional[np.ndarray]:
        """Convert depth image to point cloud using camera intrinsics"""
        try:
            # Get camera intrinsics
            fx = self.camera_intrinsics['fx']
            fy = self.camera_intrinsics['fy']
            cx = self.camera_intrinsics['cx']
            cy = self.camera_intrinsics['cy']
            
            # Create Open3D camera intrinsic
            height, width = depth_img.shape
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
            
            # Convert depth image to Open3D format
            depth_o3d = o3d.geometry.Image(depth_img.astype(np.float32))
            
            # Generate point cloud
            point_cloud_o3d = o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d, intrinsic)
            
            # Convert to numpy array
            points = np.asarray(point_cloud_o3d.points)
            
            # Add dummy colors (RGB as zeros for now)
            if len(points) > 0:
                colors = np.zeros((len(points), 3))
                point_cloud = np.hstack([points, colors])
                return point_cloud
            else:
                return None
                
        except Exception as e:
            print(f"Error converting depth to point cloud: {e}")
            return None

    def _filter_by_workspace(self, point_cloud: np.ndarray) -> np.ndarray:
        """Filter point cloud by workspace bounds"""
        if self.workspace_bounds is None:
            return point_cloud
            
        # workspace_bounds should be [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        x_bounds, y_bounds, z_bounds = self.workspace_bounds
        
        mask = (
            (point_cloud[:, 0] >= x_bounds[0]) & (point_cloud[:, 0] <= x_bounds[1]) &
            (point_cloud[:, 1] >= y_bounds[0]) & (point_cloud[:, 1] <= y_bounds[1]) &
            (point_cloud[:, 2] >= z_bounds[0]) & (point_cloud[:, 2] <= z_bounds[1])
        )
        
        return point_cloud[mask]

    def _sample_point_cloud(self, point_cloud: np.ndarray, num_points: int) -> np.ndarray:
        """Sample point cloud to fixed number of points"""
        if len(point_cloud) == 0:
            return np.zeros((num_points, point_cloud.shape[1] if point_cloud.ndim > 1 else 3))
        
        if len(point_cloud) <= num_points:
            return point_cloud
            
        if self.point_cloud_sampling_method == 'uniform':
            # Uniform sampling
            indices = np.random.choice(len(point_cloud), num_points, replace=False)
            return point_cloud[indices]
        elif self.point_cloud_sampling_method == 'fps':
            # Farthest point sampling
            try:
                points_tensor = torch.from_numpy(point_cloud[:, :3]).unsqueeze(0).float()
                if torch.cuda.is_available():
                    points_tensor = points_tensor.cuda()
                
                _, sampled_indices = torch3d_ops.sample_farthest_points(
                    points=points_tensor, K=torch.tensor([num_points]))
                
                sampled_indices = sampled_indices.squeeze(0).cpu().numpy()
                return point_cloud[sampled_indices]
            except Exception as e:
                print(f"FPS sampling failed, using uniform: {e}")
                indices = np.random.choice(len(point_cloud), num_points, replace=False)
                return point_cloud[indices]
        else:
            raise ValueError(f"Unknown sampling method: {self.point_cloud_sampling_method}")

    def get_validation_dataset(self):
        """Get validation dataset"""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """Get normalizer for the dataset"""
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """Convert sample to data format"""
        agent_pos = sample['state'].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud,  # T, num_points, 6 (xyz + rgb)
                'agent_pos': agent_pos,      # T, D_pos
            },
            'action': sample['action'].astype(np.float32)  # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset"""
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data