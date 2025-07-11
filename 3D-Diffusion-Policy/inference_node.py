#!/usr/bin/env python3

import cv2
import numpy as np
import threading
import time
import torch
import argparse
import os
import sys
import copy
import boto3
import json
from omegaconf import OmegaConf
from termcolor import cprint
import hydra
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from franky import Affine, Robot ,JointMotion

# Add the project path
sys.path.append('/mnt/data/Diffusion3d/3D-Diffusion-Policy')

# Import the diffusion policy components
from train_modified import TrainDP3Workspace
from diffusion_policy_3d.dataset.franka_dataset import FrankaDataset
from intelrealsense import IntelRealSenseCamera, IntelRealSenseCameraConfig

FREQUENCY = 60.0  # Inference frequency (Hz)

class FrankaDiffusionInference:
    def __init__(self, 
                 config_path: str,
                 cameras_config_path: str,
                 robot_ip: str = "172.16.0.2",
                 s3_bucket: str = "pr-checkpoints",
                 latest: bool = True,
                 epoch: int = None,
                 device: str = "cuda",
                 use_ema: bool = False,
                 restore_checkpoint: bool = True,
                 workspace_bounds: list = None):
        
        self.device = device
        self.use_ema = use_ema
        self.cameras_config_path = cameras_config_path
        self.robot_ip = robot_ip
        
        # Workspace bounds for point cloud cropping
        if workspace_bounds is None:
            self.workspace_bounds = [
                [-0.1, 0.8],  # X bounds (meters)
                [-0.4, 0.5],  # Y bounds (meters)  
                [-0.2, 0.8]    # Z bounds (meters)
            ]
        else:
            self.workspace_bounds = workspace_bounds
        
        # Robot state
        self.joint_state = None
        self.gripper_state = None
        self.lock = threading.Lock()
        
        # Observation history for model input
        self.observation_history = []
        self.max_history_length = 10  # Will be set from config
        
        # Initialize Franka robot
        self.setup_robot()
        
        # Load camera extrinsics and setup cameras
        self.load_camera_config()
        self.setup_cameras()
        
        # Load model
        self.load_model(config_path, s3_bucket, latest, epoch, restore_checkpoint)
        
        print("Franka Diffusion Policy inference node started.")
        
    def setup_robot(self):
        """Initialize connection to Franka robot using Franky"""
        try:
            self.robot = Robot(self.robot_ip)
            print(f"Connected to Franka robot at {self.robot_ip}")
            
            
        except Exception as e:
            print(f"Failed to connect to Franka robot: {e}")
            raise
            
    def get_robot_state(self):
        """Get current robot joint states and gripper position"""
        try:
            # Get joint positions
            robot_state = self.robot.state
            joint_positions = robot_state.q  # Joint positions [7]
            
            # Get gripper position (assuming you have gripper setup)
            # gripper_width = self.robot.gripper.width()  # Uncomment if gripper available
            gripper_width = 0.00  # Default gripper width for now
            
            with self.lock:
                self.joint_state = joint_positions.tolist()
                self.gripper_state = [gripper_width]
                
            return True
            
        except Exception as e:
            print(f"Failed to read robot state: {e}")
            return False
        
    def load_camera_config(self):
        """Load camera configuration (intrinsics and extrinsics) from JSON file"""
        try:
            with open(self.cameras_config_path, 'r') as f:
                camera_data = json.load(f)
            
            self.extrinsics = {}
            self.camera_intrinsics = {}
            
            for camera_name, camera_config in camera_data.items():
                # Load extrinsics (transformation from camera to base frame)
                if 'extrinsics' in camera_config:
                    extrinsics = camera_config['extrinsics']
                    if 'SE3' in extrinsics:
                        # Use SE3 4x4 matrix directly
                        transform = np.array(extrinsics['SE3'])
                        self.extrinsics[camera_name] = transform
                    else:
                        # Fallback: try to load as [x, y, z, qx, qy, qz, qw]
                        extrinsic_data = extrinsics
                        if isinstance(extrinsic_data, list) and len(extrinsic_data) == 7:
                            translation = np.array(extrinsic_data[:3])
                            quaternion = np.array(extrinsic_data[3:7])  # [qx, qy, qz, qw]
                            rotation_matrix = R.from_quat(quaternion).as_matrix()
                            transform = np.eye(4)
                            transform[:3, :3] = rotation_matrix
                            transform[:3, 3] = translation
                            self.extrinsics[camera_name] = transform
                # Load intrinsics
                if 'intrinsics' in camera_config:
                    intrinsics = camera_config['intrinsics']
                    self.camera_intrinsics[camera_name] = {
                        'fx': intrinsics['fx'],
                        'fy': intrinsics['fy'],
                        'cx': intrinsics['cx'],
                        'cy': intrinsics['cy']
                    }
            
            print(f"Loaded camera config for {len(self.extrinsics)} cameras")
            
        except Exception as e:
            print(f"Failed to load camera config: {e}")
            # Use default extrinsics if config loading fails
            self.extrinsics = {}
            self.camera_intrinsics = {}
        
    def setup_cameras(self):
        """Setup Intel RealSense cameras for left and right only (merged_1024)"""
        self.camera_configs = {
            "left_camera": IntelRealSenseCameraConfig(
                serial_number="142422250807",
                fps=30, 
                width=640, 
                height=480,
                use_depth=True,
                mock=False
            ),
            "right_camera": IntelRealSenseCameraConfig(
                serial_number="025522060843",  
                fps=30, 
                width=640, 
                height=480,
                use_depth=True,
                mock=False,
                rotation=90
            ),
        }
        
        self.cameras = {}
        for name, cfg in self.camera_configs.items():
            try:
                self.cameras[name] = IntelRealSenseCamera(cfg)
                self.cameras[name].connect()
                print(f"Connected to camera {name}")
            except Exception as e:
                print(f"Failed to connect camera {name}: {e}")
                
    def load_model(self, config_path, s3_bucket, latest, epoch, restore_checkpoint):
        """Load the 3D Diffusion Policy model"""
        try:
            # Get checkpoint path
            if latest:
                checkpoint_name = "checkpoint_latest.pth"
            elif epoch is not None:
                checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
            else:
                raise ValueError("Either latest must be True or epoch must be specified")
            
            # Load config using hydra's compose API
            config_path = os.path.abspath(config_path)
            config_dir = os.path.dirname(config_path)
            config_name = os.path.basename(config_path)
            
            hydra.initialize(config_path=os.path.relpath(config_dir, start=os.getcwd()), version_base=None)
            self.cfg = hydra.compose(config_name=config_name)
            
            # Set observation history length from config
            self.max_history_length = self.cfg.n_obs_steps
            
            # Create workspace with model
            self.workspace = TrainDP3Workspace(self.cfg)
            
            # Load checkpoint from S3 or local cache
            local_path = checkpoint_name
            
            if s3_bucket and s3_bucket.lower() != "null":
                s3_client = boto3.client('s3')
                if latest:
                    s3_path = "DP3_outputs/latest/checkpoint.pth"
                else:
                    s3_path = f"DP3_outputs/epoch_{epoch}/checkpoint.pth"
                
                if restore_checkpoint and os.path.exists(local_path):
                    print(f"Loading checkpoint from local cache: {local_path}")
                else:
                    print(f"Downloading checkpoint from S3: {s3_bucket}/{s3_path}")
                    s3_client.download_file(s3_bucket, s3_path, local_path)
            else:
                print(f"S3 bucket not provided, loading checkpoint locally from: {local_path}")
                if not os.path.exists(local_path):
                    raise FileNotFoundError(f"Local checkpoint file not found: {local_path}")
            
            # Load checkpoint
            ckpt = torch.load(local_path, map_location=self.device, weights_only=False)
            self.workspace.model.load_state_dict(ckpt['model_state_dict'])
            
            # Load EMA model if available
            if self.workspace.ema_model is not None and 'ema_state_dict' in ckpt:
                self.workspace.ema_model.load_state_dict(ckpt['ema_state_dict'])
            
            # Load normalizer state (essential for proper inference)
            if hasattr(self.workspace.model, 'normalizer') and 'normalizer_state_dict' in ckpt:
                self.workspace.model.normalizer.load_state_dict(ckpt['normalizer_state_dict'])
            
            # Move models to device
            self.workspace.model.to(self.device)
            if self.workspace.ema_model is not None:
                self.workspace.ema_model.to(self.device)
                
            # Set model to eval mode
            self.workspace.model.eval()
            if self.workspace.ema_model is not None:
                self.workspace.ema_model.eval()
            
            print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
    def get_current_images_and_point_clouds(self):
        """Get current images and point clouds from left and right cameras only"""
        images = {}
        point_clouds = []
        
        # Only process left and right cameras for merged_1024
        target_cameras = ["left_camera", "right_camera"]
        
        for name in target_cameras:
            if name in self.cameras:
                try:
                    # Read color and depth from camera
                    color_img, depth_map = self.cameras[name].read()
                    
                    if color_img is not None and depth_map is not None:
                        images[name] = color_img.copy()
                        
                        # Convert depth to point cloud using RealSense
                        pcd = self.depth_to_point_cloud_rs(color_img, depth_map, name)
                        if pcd is not None and len(pcd) > 0:
                            point_clouds.append(pcd)
                            
                except Exception as e:
                    print(f"Failed to read from camera {name}: {e}")
                
        return images, point_clouds
        
    def depth_to_point_cloud_rs(self, color_img, depth_map, camera_name):
        """Convert depth map to point cloud using RealSense and transform to world frame"""
        try:
            # Get camera intrinsics
            if camera_name in self.camera_intrinsics:
                intrinsics = self.camera_intrinsics[camera_name]
                fx, fy = intrinsics['fx'], intrinsics['fy']
                cx, cy = intrinsics['cx'], intrinsics['cy']
            else:
                # Default intrinsics if not found
                fx = fy = 525.0
                cx = depth_map.shape[1] / 2
                cy = depth_map.shape[0] / 2
                print(f"Using default intrinsics for {camera_name}")
            
            # Create RealSense intrinsics object
            height, width = depth_map.shape
            rs_intrinsics = rs.intrinsics()
            rs_intrinsics.width = width
            rs_intrinsics.height = height
            rs_intrinsics.fx = fx
            rs_intrinsics.fy = fy
            rs_intrinsics.ppx = cx
            rs_intrinsics.ppy = cy
            rs_intrinsics.model = rs.distortion.brown_conrady
            rs_intrinsics.coeffs = [0, 0, 0, 0, 0]  # No distortion
            
            # Manual point cloud generation (RealSense way)
            points = []
            for y in range(height):
                for x in range(width):
                    depth_value = depth_map[y, x]
                    if depth_value > 0:  # Valid depth
                        # Convert to 3D point using RealSense deproject
                        point = rs.rs2_deproject_pixel_to_point(rs_intrinsics, [x, y], depth_value / 1000.0)
                        points.append(point)
            
            if len(points) == 0:
                return None
                
            # Convert to numpy array
            v = np.array(points)
            
            # Filter out far points
            valid_mask = (v[:, 2] > 0) & (v[:, 2] < 5.0)
            v = v[valid_mask]
            
            if len(v) == 0:
                return None
            
            # Transform to world frame using extrinsics
            if camera_name in self.extrinsics:
                transform = self.extrinsics[camera_name]
                # Add homogeneous coordinate
                v_homogeneous = np.hstack([v, np.ones((v.shape[0], 1))])
                # Transform points
                v_transformed = (transform @ v_homogeneous.T).T
                v = v_transformed[:, :3]
            else:
                print(f"No extrinsics found for {camera_name}, using camera frame")
            
            return v
            
        except Exception as e:
            print(f"Failed to convert depth to point cloud with RealSense: {e}")
            return None
            
    def crop_workspace(self, points):
        """Crop point cloud to workspace bounds"""
        if len(points) == 0:
            return points
            
        # Apply workspace bounds
        mask = (
            (points[:, 0] >= self.workspace_bounds[0][0]) & 
            (points[:, 0] <= self.workspace_bounds[0][1]) &
            (points[:, 1] >= self.workspace_bounds[1][0]) & 
            (points[:, 1] <= self.workspace_bounds[1][1]) &
            (points[:, 2] >= self.workspace_bounds[2][0]) & 
            (points[:, 2] <= self.workspace_bounds[2][1])
        )
        
        return points[mask]
    
    def farthest_point_sampling(self, points, num_points):
        """Farthest Point Sampling (FPS) to downsample point cloud"""
        if len(points) <= num_points:
            return points
            
        # Simple FPS implementation
        sampled_indices = [0]  # Start with first point
        distances = np.full(len(points), np.inf)
        
        for _ in range(num_points - 1):
            # Update distances to nearest sampled point
            last_idx = sampled_indices[-1]
            last_point = points[last_idx]
            new_distances = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, new_distances)
            
            # Select point farthest from all sampled points
            next_idx = np.argmax(distances)
            sampled_indices.append(next_idx)
        
        return points[sampled_indices]
            
    def merge_point_clouds(self, point_clouds):
        """Merge multiple point clouds, crop workspace, and FPS to 1024 points"""
        if not point_clouds:
            return None
            
        try:
            # Concatenate all point clouds
            merged_pcd = np.concatenate(point_clouds, axis=0)
            
            # Crop to workspace bounds
            cropped_pcd = self.crop_workspace(merged_pcd)
            
            if len(cropped_pcd) == 0:
                print("No points left after workspace cropping")
                return np.zeros((1024, 3))  # Return dummy points
            
            # Target is 1024 points for merged_1024 dataset type
            target_points = 1024
            
            if len(cropped_pcd) > target_points:
                # Use Farthest Point Sampling
                if hasattr(self.cfg.task.dataset, 'point_cloud_sampling_method') and \
                   self.cfg.task.dataset.point_cloud_sampling_method == 'fps':
                    sampled_pcd = self.farthest_point_sampling(cropped_pcd, target_points)
                else:
                    # Random sampling as fallback
                    indices = np.random.choice(len(cropped_pcd), target_points, replace=False)
                    sampled_pcd = cropped_pcd[indices]
            elif len(cropped_pcd) < target_points:
                # Pad by repeating points
                padding_needed = target_points - len(cropped_pcd)
                repeat_indices = np.random.choice(len(cropped_pcd), padding_needed, replace=True)
                padding_points = cropped_pcd[repeat_indices]
                sampled_pcd = np.concatenate([cropped_pcd, padding_points], axis=0)
            else:
                sampled_pcd = cropped_pcd
                
            return sampled_pcd
            
        except Exception as e:
            print(f"Failed to merge point clouds: {e}")
            return None
            
    def create_observation(self):
        """Create observation dictionary from current sensor data"""
        with self.lock:
            if self.joint_state is None or self.gripper_state is None:
                return None
                
            # Get current agent position (joint + gripper states)
            agent_pos = self.joint_state + self.gripper_state  # [7 + 1 = 8]
            
        # Get images and point clouds
        images, point_clouds = self.get_current_images_and_point_clouds()
        
        # Merge point clouds
        merged_pcd = self.merge_point_clouds(point_clouds)
        
        if merged_pcd is None:
            return None
            
        # Create observation dictionary
        obs = {
            'point_cloud': merged_pcd,  # [N, 3]
            'agent_pos': np.array(agent_pos),  # [8]
            'images': images  # For debugging/visualization
        }
        
        return obs
        
    def update_observation_history(self, obs):
        """Update observation history with new observation"""
        self.observation_history.append(obs)
        
        # Keep only the required number of observations
        if len(self.observation_history) > self.max_history_length:
            self.observation_history = self.observation_history[-self.max_history_length:]
            
    def create_model_input(self):
        """Create model input from observation history"""
        if len(self.observation_history) == 0:
            return None
            
        # Pad history if needed
        required_length = self.max_history_length
        current_length = len(self.observation_history)
        
        if current_length < required_length:
            # Repeat the first observation to pad
            padding_needed = required_length - current_length
            first_obs = self.observation_history[0]
            padded_history = [first_obs] * padding_needed + self.observation_history
        else:
            padded_history = self.observation_history[-required_length:]
            
        # Stack observations
        point_clouds = []
        agent_poses = []
        
        for obs in padded_history:
            point_clouds.append(obs['point_cloud'])
            agent_poses.append(obs['agent_pos'])
            
        # Convert to tensors
        point_cloud_tensor = torch.from_numpy(np.stack(point_clouds)).float()  # [T, N, 3]
        agent_pos_tensor = torch.from_numpy(np.stack(agent_poses)).float()     # [T, 8]
        
        # Add batch dimension
        point_cloud_tensor = point_cloud_tensor.unsqueeze(0).to(self.device)  # [1, T, N, 3]
        agent_pos_tensor = agent_pos_tensor.unsqueeze(0).to(self.device)      # [1, T, 8]
        
        obs_dict = {
            'point_cloud': point_cloud_tensor,
            'agent_pos': agent_pos_tensor
        }
        
        return obs_dict
        
    def run_inference(self, obs_dict):
        """Run inference with the diffusion policy model"""
        try:
            # Select the appropriate model
            if self.use_ema and self.workspace.ema_model is not None:
                policy = self.workspace.ema_model
            else:
                policy = self.workspace.model
                
            # Run inference
            with torch.no_grad():
                result = policy.predict_action(obs_dict)
                pred_action = result['action_pred']  # [1, horizon, 8]
                
            return pred_action.cpu().numpy()
            
        except Exception as e:
            print(f"Inference failed: {e}")
            return None
            
    def send_actions(self, actions):
        """Send predicted actions to Franka robot"""
        try:
            # Take the first action from the predicted sequence
            action = actions[0, 0, :]  # [8] - first batch, first timestep
            
            joint_actions = action[:7]  # First 7 elements are joint actions
            gripper_action = action[7]  # Last element is gripper action
            
            # Send joint commands to robot
            try:
                # Move to joint position
                print(f"Moving robot to joint positions: {joint_actions.tolist()}")
                self.robot.move(JointMotion(joint_actions.tolist() , relative_dynamics_factor=0.01))  # Example relative dynamics factor
                
                # Control gripper (if available)
                # if hasattr(self.robot, 'gripper'):
                #     self.robot.gripper.move(gripper_action)
                
                print(f"Sent actions - Joints: {joint_actions}, Gripper: {gripper_action}")
                
            except Exception as e:
                print(f"Failed to send joint commands: {e}")
                
        except Exception as e:
            print(f"Failed to process actions: {e}")
            
    def save_point_cloud(self, point_cloud, filename=None):
        """Save point cloud to file for debugging/visualization"""
        try:
            if filename is None:
                filename = f"point_cloud_{int(time.time())}.npy"
            
            np.save(filename, point_cloud)
            print(f"Saved point cloud to {filename}")
            
        except Exception as e:
            print(f"Failed to save point cloud: {e}")
            
    def inference_step(self):
        """Main inference loop"""
        try:
            # Read current robot state
            if not self.get_robot_state():
                return
                
            # Create observation from current sensor data
            obs = self.create_observation()
            if obs is None:
                return
                
            # Update observation history
            self.update_observation_history(obs)
            
            # Create model input
            obs_dict = self.create_model_input()
            if obs_dict is None:
                return
                
            # Run inference
            actions = self.run_inference(obs_dict)
            if actions is None:
                return
                
            # Send actions to robot
            self.send_actions(actions)
            
            # Save point cloud for debugging (optional)
            # self.save_point_cloud(obs['point_cloud'])
            
        except Exception as e:
            print(f"Error in inference step: {e}")
            
    def run(self):
        """Run the inference loop"""
        try:
            print("Starting inference loop...")
            
            while True:
                start_time = time.time()
                
                # Run inference step
                self.inference_step()
                
                # Maintain target frequency
                elapsed_time = time.time() - start_time
                print(f"Elapsed time for step: {elapsed_time:.3f} seconds")
                sleep_time = max(0, (1.0 / FREQUENCY) - elapsed_time)
                # time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("Stopping inference loop...")
        except Exception as e:
            print(f"Error in inference loop: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Disconnect cameras
            for cam in self.cameras.values():
                try:
                    cam.disconnect()
                except Exception as e:
                    print(f"Failed to disconnect camera: {e}")
                    
            # Stop robot (optional - robot will maintain last position)
            # self.robot.stop()
            
            print("Cleanup completed")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main(args=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Franka 3D Diffusion Policy Inference")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the config file used for training")
    parser.add_argument("--cameras_config_path", type=str, required=True,
                        help="Path to cameras.json file with intrinsics and extrinsics")
    parser.add_argument("--robot_ip", type=str, default="172.16.0.2",
                        help="IP address of the Franka robot")
    parser.add_argument("--s3_bucket", type=str, default="pr-checkpoints",
                        help="S3 bucket containing the checkpoint")
    parser.add_argument("--latest", action="store_true", default=True,
                        help="Use the latest checkpoint")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific epoch checkpoint to load")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--use_ema", action="store_true",
                        help="Use EMA model for inference")
    parser.add_argument("--restore_checkpoint", action="store_true", default=True,
                        help="Restore from a local checkpoint if it exists")
    parser.add_argument("--workspace_bounds", nargs=6, type=float, default=None,
                        help="Workspace bounds as x_min x_max y_min y_max z_min z_max")
    
    parsed_args = parser.parse_args(args)
    
    # Parse workspace bounds
    workspace_bounds = None
    if parsed_args.workspace_bounds:
        workspace_bounds = [
            [parsed_args.workspace_bounds[0], parsed_args.workspace_bounds[1]],
            [parsed_args.workspace_bounds[2], parsed_args.workspace_bounds[3]],
            [parsed_args.workspace_bounds[4], parsed_args.workspace_bounds[5]]
        ]
    
    # Create and run the inference system
    inference_system = FrankaDiffusionInference(
        config_path=parsed_args.config_path,
        cameras_config_path=parsed_args.cameras_config_path,
        robot_ip=parsed_args.robot_ip,
        s3_bucket=parsed_args.s3_bucket,
        latest=parsed_args.latest,
        epoch=parsed_args.epoch,
        device=parsed_args.device,
        use_ema=parsed_args.use_ema,
        restore_checkpoint=parsed_args.restore_checkpoint,
        workspace_bounds=workspace_bounds
    )
    
    # Run the inference loop
    inference_system.run()


if __name__ == "__main__":
    main()
