#!/usr/bin/env python3

import socket
import pickle
import cv2
import numpy as np
import threading
import time
import json
import argparse
import os
import sys
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from franky import Affine, Robot, JointMotion

# Add the project path
sys.path.append('/mnt/data/Diffusion3d/3D-Diffusion-Policy')

# Import camera components
from intelrealsense import IntelRealSenseCamera, IntelRealSenseCameraConfig

FREQUENCY = 60.0  # Inference frequency (Hz)

class FrankaDiffusionClient:
    def __init__(self, 
                 cameras_config_path: str,
                 server_ip: str,
                 server_port: int = 5000,
                 robot_ip: str = "172.16.0.2",
                 workspace_bounds: list = None,
                 max_history_length: int = 2):
        
        self.cameras_config_path = cameras_config_path
        self.server_ip = server_ip
        self.server_port = server_port
        self.robot_ip = robot_ip
        self.max_history_length = max_history_length
        
        # Workspace bounds for point cloud cropping - MATCH TRAINING DATA EXACTLY
        if workspace_bounds is None:
            # Use exact same bounds as training config (dp3.yaml -> task/franka_custom.yaml)
            self.workspace_bounds = [
                [-0.1, 0.8],   # x: forward/backward from robot base
                [-0.35, 0.3],   # y: left/right from robot base  
                [-0.1, 0.8]    # z: up/down from robot base
            ]
        else:
            self.workspace_bounds = workspace_bounds
        
        # Robot state
        self.joint_state = None
        self.gripper_state = None
        self.lock = threading.Lock()
        
        # Observation history for model input
        self.observation_history = []
        
        # Initialize Franka robot
        self.setup_robot()
        
        # Load camera extrinsics and setup cameras
        self.load_camera_config()
        self.setup_cameras()
        
        print("Franka Diffusion Policy client started.")
        
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
            gripper_width = 0.04  # Default gripper width for now
            
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
        """Setup Intel RealSense cameras - MATCH RECORDING SCRIPT EXACTLY"""
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
                mock=False
            ),
        }
        
        self.cameras = {}
        for name, cfg in self.camera_configs.items():
            try:
                self.cameras[name] = IntelRealSenseCamera(cfg)
                self.cameras[name].connect()
                self.cameras[name].async_read()  # Start async reading like recording
                print(f"Connected to camera {name} (serial: {cfg.serial_number})")
            except Exception as e:
                print(f"Failed to connect camera {name}: {e}")
                
    def get_current_images_and_point_clouds(self):
        """Get current images and point clouds from left and right cameras - MATCH RECORDING"""
        images = {}
        point_clouds = []
        
        # Only process left and right cameras for merged_1024 (same as training)
        target_cameras = ["left_camera", "right_camera"]
        
        for name in target_cameras:
            if name in self.cameras:
                try:
                    # Read color and depth from camera - SAME METHOD AS RECORDING
                    color_img, depth_map = self.cameras[name].async_read()
                    
                    if color_img is not None and depth_map is not None:
                        images[name] = color_img.copy()
                        
                        # Convert depth to point cloud using EXACT SAME METHOD as training conversion
                        pcd = self.depth_to_point_cloud_rs(color_img, depth_map, name)
                        if pcd is not None and len(pcd) > 0:
                            point_clouds.append(pcd)
                            
                except Exception as e:
                    print(f"Failed to read from camera {name}: {e}")
                
        return images, point_clouds
        
    def depth_to_point_cloud_rs(self, color_img, depth_map, camera_name):
        """Convert depth map to point cloud using same method as training data conversion"""
        try:
            # Get camera intrinsics - match training data exactly
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
            
            # Get image dimensions
            height, width = depth_map.shape
            
            # Create coordinate grids - same as training conversion
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            
            # Convert depth to meters - MATCH TRAINING DATA SCALING EXACTLY
            if 'wrist' in camera_name.lower():
                depth_m = depth_map.astype(np.float32) / 10000.0  # Wrist camera scaling
            else:
                depth_m = depth_map.astype(np.float32) / 1000.0   # Left/right camera scaling
            
            # Remove invalid depth values - same thresholds as training
            valid_mask = (depth_m > 0) & (depth_m < 5.0)
            
            if np.sum(valid_mask) == 0:
                print(f"No valid depth pixels for {camera_name}")
                return None
            
            # Convert to 3D points in camera frame - SAME METHOD AS TRAINING
            x = (u - cx) * depth_m / fx
            y = (v - cy) * depth_m / fy
            z = depth_m
            
            # Apply valid mask
            x = x[valid_mask]
            y = y[valid_mask]
            z = z[valid_mask]
            
            # Stack XYZ coordinates
            points_3d = np.stack([x, y, z], axis=1)
            
            # Transform to world frame using extrinsics - same as training
            if camera_name in self.extrinsics:
                transform = self.extrinsics[camera_name]
                # Add homogeneous coordinate
                ones = np.ones((points_3d.shape[0], 1))
                points_homogeneous = np.hstack([points_3d, ones])
                # Transform points
                points_transformed = (transform @ points_homogeneous.T).T
                points_3d = points_transformed[:, :3]  # Remove homogeneous coordinate
            else:
                print(f"No extrinsics found for {camera_name}, using camera frame")
            
            return points_3d
            
        except Exception as e:
            print(f"Failed to convert depth to point cloud: {e}")
            import traceback
            traceback.print_exc()
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
        """Farthest Point Sampling (FPS) to downsample point cloud - match training exactly"""
        if len(points) <= num_points:
            # If we have fewer points than desired, pad with duplicates
            if len(points) == 0:
                return np.zeros((num_points, 3))
            padding_needed = num_points - len(points)
            repeat_indices = np.random.choice(len(points), padding_needed, replace=True)
            padding_points = points[repeat_indices]
            return np.concatenate([points, padding_points], axis=0)
            
        try:
            # Try PyTorch3D FPS (same as training)
            import torch
            import pytorch3d.ops as torch3d_ops
            
            xyz_tensor = torch.from_numpy(points[:, :3]).float()
            sampled_points, indices = torch3d_ops.sample_farthest_points(
                points=xyz_tensor.unsqueeze(0), 
                K=[num_points]
            )
            indices = indices.squeeze(0).numpy()
            return points[indices]
            
        except (ImportError, Exception) as e:
            # Fallback to random sampling if PyTorch3D not available or fails
            print(f"PyTorch3D not available or failed ({e}), using random sampling")
            indices = np.random.choice(len(points), num_points, replace=False)
            return points[indices]
            
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
            
            # Target is exactly 1024 points to match training
            target_points = 1024
            
            if len(cropped_pcd) > target_points:
                # Use Farthest Point Sampling - consistent with training
                sampled_pcd = self.farthest_point_sampling(cropped_pcd, target_points)
            elif len(cropped_pcd) < target_points:
                # Pad by repeating points - consistent with training
                padding_needed = target_points - len(cropped_pcd)
                repeat_indices = np.random.choice(len(cropped_pcd), padding_needed, replace=True)
                padding_points = cropped_pcd[repeat_indices]
                sampled_pcd = np.concatenate([cropped_pcd, padding_points], axis=0)
            else:
                sampled_pcd = cropped_pcd
                
            # Ensure we have exactly 1024 points
            assert sampled_pcd.shape[0] == 1024, f"Expected 1024 points, got {sampled_pcd.shape[0]}"
            
            return sampled_pcd
            
        except Exception as e:
            print(f"Failed to merge point clouds: {e}")
            import traceback
            traceback.print_exc()
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
        
        # Debug: save point cloud for debugging (commented out for performance)
        if merged_pcd is not None:
            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(merged_pcd)
                o3d.io.write_point_cloud("merged_point_cloud.pcd", pcd)
                print("Saved merged point cloud to 'merged_point_cloud.pcd'")
            except ImportError:
                pass
        
        if merged_pcd is None:
            return None
        
        # Ensure point clouds are [N, 3] format (remove color if present)
        if merged_pcd.shape[1] == 6:  # [x, y, z, r, g, b]
            merged_pcd = merged_pcd[:, :3]
            
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
            
        # Pad history if needed - use exactly the required history length
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
            
        # Convert to numpy arrays - match training data format exactly
        point_cloud_array = np.stack(point_clouds)  # [T, N, 3]
        agent_pos_array = np.stack(agent_poses)     # [T, D]
        
        
        # Ensure point clouds are exactly the right shape [T, 1024, 3]
        if point_cloud_array.shape[1] != 1024:
            print(f"Warning: point cloud has {point_cloud_array.shape[1]} points, expected 1024")
        
        # Add batch dimension to match training format
        point_cloud_array = np.expand_dims(point_cloud_array, axis=0)  # [1, T, N, 3]
        agent_pos_array = np.expand_dims(agent_pos_array, axis=0)      # [1, T, D]
        
        #print agent_pos_array
        print(f"Observation shapes - point_cloud: {point_cloud_array.shape}, agent_pos: {agent_pos_array.shape}")
        #print agent_pos_array
        print(f"Observation data - agent_pos: {agent_pos_array}")
        obs_dict = {
            'point_cloud': point_cloud_array,
            'agent_pos': agent_pos_array
        }
        
        return obs_dict
        
    def send_to_server(self, obs_dict):
        """Send observation to server and receive actions"""
        try:
            # Serialize observation
            data = pickle.dumps(obs_dict) + b"<END>"
            
            # Send to server
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10.0)  # 10 second timeout
            s.connect((self.server_ip, self.server_port))
            s.sendall(data)
            
            # Receive response
            response = b""
            while True:
                packet = s.recv(4096)
                if not packet:
                    break
                response += packet
                if response.endswith(b"<END>"):
                    response = response[:-5]
                    break
            
            actions = pickle.loads(response)
            s.close()
            
            
            return actions
            
        except Exception as e:
            print(f"Failed to communicate with server: {e}")
            return None
            
    def send_actions(self, actions):
        """Send predicted actions to Franka robot"""
        try:
            if actions is None:
                print("No actions to send")
                return
                
            # Take the first action from the predicted sequence
            action = actions[0, 0, :]  # [action_dim] - first batch, first timestep
            
            # Ensure we have the expected action dimension
            if len(action) < 7:
                print(f"Warning: action has {len(action)} dimensions, expected at least 7")
                return
                
            joint_actions = action[:7]  # First 7 elements are joint actions
            gripper_action = action[7] if len(action) > 7 else 0.0  # Last element is gripper action
            
            # Send joint commands to robot
            try:
                # Move to joint position with appropriate dynamics
                print(f"Moving robot to joint positions: {joint_actions.tolist()}")
                self.robot.move(JointMotion(joint_actions.tolist(), relative_dynamics_factor=0.05))
                
                # Control gripper (if available)
                # if hasattr(self.robot, 'gripper'):
                #     self.robot.gripper.move(gripper_action)
                
                print(f"Sent actions - Joints: {joint_actions}, Gripper: {gripper_action}")
                
            except Exception as e:
                print(f"Failed to send joint commands: {e}")
                
        except Exception as e:
            print(f"Failed to process actions: {e}")
            import traceback
            traceback.print_exc()
            
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
                print("Failed to get robot state, skipping inference step")
                return
                
            # Create observation from current sensor data
            obs = self.create_observation()
            if obs is None:
                print("Failed to create observation, skipping inference step")
                return
                
            # Validate observation shapes
            if obs['point_cloud'].shape[0] != 1024:
                print(f"Warning: Point cloud has {obs['point_cloud'].shape[0]} points, expected 1024")
            if obs['agent_pos'].shape[0] != 8:
                print(f"Warning: Agent pos has {obs['agent_pos'].shape[0]} dimensions, expected 8")
                
            # Update observation history
            self.update_observation_history(obs)
            
            # Create model input
            obs_dict = self.create_model_input()
            if obs_dict is None:
                print("Failed to create model input")
                return
                
            # Validate model input shapes
            print(f"Model input shapes - point_cloud: {obs_dict['point_cloud'].shape}, agent_pos: {obs_dict['agent_pos'].shape}")
                
            # Send to server for inference
            actions = self.send_to_server(obs_dict)
            if actions is None:
                print("No actions received from server")
                return
            print(f"Received actions: {actions.shape}")
            
            # Send actions to robot
            self.send_actions(actions)
            
            # Save point cloud for debugging (optional)
            # self.save_point_cloud(obs['point_cloud'])
            
        except Exception as e:
            print(f"Error in inference step: {e}")
            import traceback
            traceback.print_exc()
            
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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Franka 3D Diffusion Policy Client")
    parser.add_argument("--cameras_config_path", type=str, required=True,
                        help="Path to cameras.json file with intrinsics and extrinsics")
    parser.add_argument("--server_ip", type=str, required=True,
                        help="IP address of the inference server")
    parser.add_argument("--server_port", type=int, default=5000,
                        help="Port of the inference server")
    parser.add_argument("--robot_ip", type=str, default="172.16.0.2",
                        help="IP address of the Franka robot")
    parser.add_argument("--workspace_bounds", nargs=6, type=float, default=None,
                        help="Workspace bounds as x_min x_max y_min y_max z_min z_max")
    parser.add_argument("--max_history_length", type=int, default=2,
                        help="Maximum length of observation history")
    
    args = parser.parse_args()
    
    # Parse workspace bounds
    workspace_bounds = None
    if args.workspace_bounds:
        workspace_bounds = [
            [args.workspace_bounds[0], args.workspace_bounds[1]],
            [args.workspace_bounds[2], args.workspace_bounds[3]],
            [args.workspace_bounds[4], args.workspace_bounds[5]]
        ]
    
    # Create and run the client
    client = FrankaDiffusionClient(
        cameras_config_path=args.cameras_config_path,
        server_ip=args.server_ip,
        server_port=args.server_port,
        robot_ip=args.robot_ip,
        workspace_bounds=workspace_bounds,
        max_history_length=args.max_history_length
    )
    
    # Run the inference loop
    client.run()


if __name__ == "__main__":
    main()

