#!/usr/bin/env python3
"""
Convert franka teleop dataset to point clouds using RGB-D information.
This script processes each episode and converts frames to point clouds using:
- RGB and depth images from multiple cameras
- Camera extrinsics from JSON file
- RealSense camera intrinsics (estimated or from camera properties)
- Point cloud transformation to robot base frame
- Cropping within workspace bounds
- Farthest Point Sampling (FPS)
"""     

import os
import json
import numpy as np
import cv2
import open3d as o3d
import argparse
import torch
import pytorch3d.ops as torch3d_ops
from tqdm import tqdm
import pickle
from pathlib import Path
import logging
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_episode_worker(episode_name, dataset_path, cameras_path, workspace_bounds, num_points, camera_to_process=None):
    """
    Worker function for parallel episode processing.
    
    Args:
        episode_name (str): Name of the episode to process
        dataset_path (Path): Path to dataset
        cameras_path (Path): Path to cameras config
        workspace_bounds (list): Workspace bounds
        num_points (int): Number of points for FPS
        camera_to_process (str): Process only specific camera (optional)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a new converter instance for this worker
        converter = DatasetConverter(
            dataset_path=dataset_path,
            cameras_path=cameras_path,
            output_path=None,  # Not used in new version
            num_points=num_points,
            workspace_bounds=workspace_bounds,
            max_workers=1,  # Use sequential processing within worker to avoid nested parallelism
            sequential=True,  # Force sequential frame processing in workers
            camera_to_process=camera_to_process
        )
        
        # Process the episode
        converter.process_episode(episode_name)
        return True
        
    except Exception as e:
        logger.error(f"Worker failed to process {episode_name}: {e}")
        return False

class DatasetConverter:
    def __init__(self, dataset_path, cameras_path, output_path, num_points=4000, workspace_bounds=None, max_workers=None, batch_size=None, sequential=False, camera_to_process=None):
        """
        Initialize the dataset converter.
        
        Args:
            dataset_path (str): Path to the franka recordings dataset
            cameras_path (str): Path to cameras.json file containing intrinsics and extrinsics
            output_path (str): Output directory for processed data
            num_points (int): Number of points for FPS sampling
            workspace_bounds (list): [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            max_workers (int): Maximum number of parallel workers
            batch_size (int): Batch size for frame processing
            sequential (bool): Force sequential processing
            camera_to_process (str): Process only specific camera ('left', 'right', 'wrist', or None for all)
        """
        logger.info("=== Initializing DatasetConverter ===")
        
        self.dataset_path = Path(dataset_path)
        self.cameras_path = Path(cameras_path)
        self.output_path = Path(output_path) if output_path else None
        self.num_points = num_points
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.batch_size = batch_size or (self.max_workers * 2)
        self.sequential = sequential
        self.camera_to_process = camera_to_process
        
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Cameras config path: {self.cameras_path}")
        if self.output_path:
            logger.info(f"Output path: {self.output_path}")
        logger.info(f"Target points per cloud: {self.num_points}")
        if self.camera_to_process:
            logger.info(f"Processing only camera: {self.camera_to_process}")
        else:
            logger.info("Processing all cameras")
        logger.info(f"Parallelization settings:")
        logger.info(f"  - Max workers: {self.max_workers}")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Sequential mode: {self.sequential}")
        
        # Validate paths
        if not self.dataset_path.exists():
            logger.error(f"Dataset path does not exist: {self.dataset_path}")
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        if not self.cameras_path.exists():
            logger.error(f"Cameras config file does not exist: {self.cameras_path}")
            raise FileNotFoundError(f"Cameras config file not found: {self.cameras_path}")
        
        # Default workspace bounds (adjust based on your robot setup)
        if workspace_bounds is None:
            self.workspace_bounds = [
                [-0.1, 0.8],   # x: forward/backward from robot base
                [-0.35, 0.3],  # y: left/right from robot base
                [-0.1, 0.8]    # z: up/down from robot base
            ]
            logger.info("Using default workspace bounds:")
        else:
            self.workspace_bounds = workspace_bounds
            logger.info("Using custom workspace bounds:")
        
        logger.info(f"  X: [{self.workspace_bounds[0][0]:.3f}, {self.workspace_bounds[0][1]:.3f}] meters")
        logger.info(f"  Y: [{self.workspace_bounds[1][0]:.3f}, {self.workspace_bounds[1][1]:.3f}] meters")
        logger.info(f"  Z: [{self.workspace_bounds[2][0]:.3f}, {self.workspace_bounds[2][1]:.3f}] meters")
        
        # Load camera configuration (intrinsics and extrinsics)
        self.load_camera_config()
        
        logger.info("=== DatasetConverter initialized successfully ===")
        if self.camera_to_process:
            logger.info(f"PCD files will be saved for {self.camera_to_process} camera only:")
            logger.info(f"  - {self.camera_to_process}_pcd/frame_XXXXXX.pcd (4000 points)")
        else:
            logger.info("PCD files will be saved directly in each episode directory")
            logger.info("  - wrist_pcd/frame_XXXXXX.pcd (4000 points)")
            logger.info("  - merged_4000/frame_XXXXXX.pcd (left+right merged, 4000 points)") 
            logger.info("  - merged_1024/frame_XXXXXX.pcd (left+right merged, 1024 points)")
        logger.info("")
        
    def load_camera_config(self):
        """Load camera configuration (intrinsics and extrinsics) from JSON file."""
        logger.info(f"Loading camera configuration from: {self.cameras_path}")
        
        try:
            with open(self.cameras_path, 'r') as f:
                camera_data = json.load(f)
            logger.info(f"Successfully loaded camera config with {len(camera_data)} cameras")
        except Exception as e:
            logger.error(f"Failed to load camera config file: {e}")
            raise
        
        self.extrinsics = {}
        self.camera_intrinsics = {}
        
        for camera_name, camera_config in camera_data.items():
            # Extract transformation matrix (extrinsics)
            transform_matrix = np.array(camera_config['extrinsics']['SE3'])
            self.extrinsics[camera_name] = transform_matrix
            
            # Extract intrinsics
            intrinsics = camera_config['intrinsics']
            self.camera_intrinsics[camera_name] = {
                'fx': intrinsics['fx'],
                'fy': intrinsics['fy'],
                'cx': intrinsics['cx'],
                'cy': intrinsics['cy'],
                'width': intrinsics['width'],
                'height': intrinsics['height']
            }
            
            # Log detailed camera info
            serial_number = camera_config.get('serial_number', 'Unknown')
            model = camera_config.get('model', 'Unknown')
            logger.info(f"Loaded config for {camera_name}:")
            logger.info(f"  - Serial: {serial_number}, Model: {model}")
            logger.info(f"  - Intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}, cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")
            logger.info(f"  - Resolution: {intrinsics['width']}x{intrinsics['height']}")
            logger.info(f"  - Transform matrix shape: {transform_matrix.shape}")
            logger.info(f"  - Translation: [{transform_matrix[0,3]:.3f}, {transform_matrix[1,3]:.3f}, {transform_matrix[2,3]:.3f}]")
        
        logger.info(f"Total cameras loaded: {len(self.extrinsics)}")
        logger.info("Using calibrated intrinsics from camera configuration file")
    
    def extract_frame_from_video(self, video_path, frame_idx):
        """
        Extract a specific frame from a video file.
        
        Args:
            video_path (Path): Path to the video file
            frame_idx (int): Frame index to extract
            
        Returns:
            np.ndarray: Extracted frame as RGB image, or None if failed
        """
        logger.debug(f"    Extracting frame {frame_idx} from {video_path}")
        
        if not video_path.exists():
            logger.warning(f"    Video file not found: {video_path}")
            return None
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.warning(f"    Could not open video: {video_path}")
                return None
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.debug(f"    Video properties: {total_frames} frames, {fps:.2f} FPS")
            
            if frame_idx >= total_frames:
                logger.warning(f"    Frame {frame_idx} exceeds video length ({total_frames} frames)")
                cap.release()
                return None
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                logger.warning(f"    Failed to read frame {frame_idx} from video")
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logger.debug(f"    Successfully extracted frame: {frame_rgb.shape}")
            
            return frame_rgb
            
        except Exception as e:
            logger.error(f"    Error extracting frame from video: {e}")
            return None
    
    def get_video_info(self, episode_dir):
        """
        Get information about video files in the episode directory.
        
        Args:
            episode_dir (Path): Episode directory
            
        Returns:
            dict: Video information for each camera
        """
        video_info = {}
        camera_mappings = {
            'left_camera': 'left',
            'wrist_camera': 'wrist', 
            'right_camera': 'right'
        }
        
        logger.debug(f"  Scanning for video files in {episode_dir}")
        
        for camera_name, folder_prefix in camera_mappings.items():
            video_path = episode_dir / f"{folder_prefix}_camera.mp4"
            
            if video_path.exists():
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if cap.isOpened():
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        video_info[camera_name] = {
                            'path': video_path,
                            'frame_count': frame_count,
                            'fps': fps,
                            'resolution': (width, height)
                        }
                        
                        logger.debug(f"    {camera_name}: {frame_count} frames, {fps:.2f} FPS, {width}x{height}")
                        cap.release()
                    else:
                        logger.warning(f"    Could not open video: {video_path}")
                except Exception as e:
                    logger.error(f"    Error reading video {video_path}: {e}")
            else:
                logger.debug(f"    No video found for {camera_name}: {video_path}")
        
        return video_info
    
    def depth_to_pointcloud(self, depth_image, rgb_image, camera_name):
        """
        Convert depth image to point cloud using camera intrinsics.
        
        Args:
            depth_image (np.ndarray): Depth image
            rgb_image (np.ndarray): RGB image  
            camera_name (str): Name of the camera
            
        Returns:
            np.ndarray: Point cloud with shape (N, 6) [x, y, z, r, g, b]
        """
        logger.debug(f"Converting depth to point cloud for {camera_name}")
        logger.debug(f"  Input depth image shape: {depth_image.shape}, dtype: {depth_image.dtype}")
        logger.debug(f"  Input RGB image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
        
        if camera_name not in self.camera_intrinsics:
            logger.warning(f"No intrinsics found for {camera_name}, using default")
            camera_name = 'right_camera'  # fallback
        
        intrinsics = self.camera_intrinsics[camera_name]
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        
        logger.debug(f"  Using intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        
        # Get image dimensions
        height, width = depth_image.shape
        logger.debug(f"  Processing image dimensions: {width}x{height}")
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert depth to meters (assuming depth is in mm)
        if(camera_name == 'wrist_camera'):
            depth_m = depth_image.astype(np.float32) / 10000.0
        else:
            depth_m = depth_image.astype(np.float32) / 1000.0
        
        logger.debug(f"  Depth range: {np.min(depth_m):.3f}m to {np.max(depth_m):.3f}m")
        
        # Remove invalid depth values
        valid_mask = (depth_m > 0) & (depth_m < 5.0)  # Remove depths > 5m
        num_valid = np.sum(valid_mask)
        total_pixels = depth_m.size
        logger.debug(f"  Valid depth pixels: {num_valid}/{total_pixels} ({100*num_valid/total_pixels:.1f}%)")
        
        if num_valid == 0:
            logger.warning(f"  No valid depth pixels found for {camera_name}")
            return np.array([]).reshape(0, 6)
        
        # Convert to 3D points in camera frame
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m
        
        # Apply valid mask
        x = x[valid_mask]
        y = y[valid_mask] 
        z = z[valid_mask]
        
        logger.debug(f"  3D point cloud range:")
        logger.debug(f"    X: {np.min(x):.3f} to {np.max(x):.3f}m")
        logger.debug(f"    Y: {np.min(y):.3f} to {np.max(y):.3f}m")
        logger.debug(f"    Z: {np.min(z):.3f} to {np.max(z):.3f}m")
        
        # Get corresponding RGB values
        rgb_image = cv2.resize(rgb_image, (width, height))
        r = rgb_image[:, :, 0][valid_mask]
        g = rgb_image[:, :, 1][valid_mask]
        b = rgb_image[:, :, 2][valid_mask]
        
        logger.debug(f"  RGB value range: R=[{np.min(r)}, {np.max(r)}], G=[{np.min(g)}, {np.max(g)}], B=[{np.min(b)}, {np.max(b)}]")
        
        # Stack into point cloud
        points = np.stack([x, y, z, r, g, b], axis=1)
        logger.debug(f"  Generated point cloud shape: {points.shape}")
        
        return points
    
    def pose_to_transform_matrix(self, pose):
        """
        Convert pose from (x, y, z, qx, qy, qz, qw) to 4x4 transformation matrix.

        Args:
            pose (np.ndarray): Pose array with 7 elements [x, y, z, qx, qy, qz, qw]
                - x, y, z: translation
                - qx, qy, qz, qw: quaternion
        
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        if len(pose) != 7:
            logger.error(f"Expected pose with 7 elements, got {len(pose)}")
            return np.eye(4)
        
        # Extract components
        translation = pose[:3]  # [x, y, z]
        quaternion = pose[3:7]  # [qx, qy, qz, qw]

        # Create transformation matrix using quaternion
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        
        return transform
    
    def transform_pointcloud(self, points, camera_name, end_effector_pose=None):
        """
        Transform point cloud from camera frame to robot base frame.
        For wrist camera, uses per-frame end effector pose.
        
        Args:
            points (np.ndarray): Point cloud in camera frame
            camera_name (str): Name of the camera
            end_effector_pose (np.ndarray): End effector pose for wrist camera (optional)
            
        Returns:
            np.ndarray: Transformed point cloud
        """
        logger.debug(f"Transforming point cloud for {camera_name}")
        logger.debug(f"  Input points shape: {points.shape}")
        
        if camera_name not in self.extrinsics:
            logger.warning(f"No extrinsics found for {camera_name}, returning original points")
            return points
        
        # For wrist camera, compute per-frame transformation
        if camera_name == 'wrist_camera' and end_effector_pose is not None:
            logger.debug(f"  Using per-frame transformation for wrist camera")
            logger.debug(f"  End effector pose: {end_effector_pose}")
            
            # Convert end effector pose to transformation matrix (T_base_ee)
            T_base_ee = self.pose_to_transform_matrix(end_effector_pose)
            logger.debug(f"  T_base_ee translation: [{T_base_ee[0,3]:.3f}, {T_base_ee[1,3]:.3f}, {T_base_ee[2,3]:.3f}]")
            
            # Get T_ee_camera from extrinsics
            T_ee_camera = self.extrinsics[camera_name]
            logger.debug(f"  T_ee_camera translation: [{T_ee_camera[0,3]:.3f}, {T_ee_camera[1,3]:.3f}, {T_ee_camera[2,3]:.3f}]")
            
            # Compute final transformation: T_base_camera = T_base_ee @ T_ee_camera
            transform_matrix = T_base_ee @ T_ee_camera
            logger.debug(f"  Final T_base_camera translation: [{transform_matrix[0,3]:.3f}, {transform_matrix[1,3]:.3f}, {transform_matrix[2,3]:.3f}]")
            
        else:
            # For fixed cameras, use static transformation from extrinsics
            transform_matrix = self.extrinsics[camera_name]
            if camera_name == 'wrist_camera':
                logger.warning(f"  Wrist camera but no end effector pose provided, using static transformation")
            else:
                logger.debug(f"  Using static transformation for {camera_name}")
            logger.debug(f"  Transform matrix translation: [{transform_matrix[0,3]:.3f}, {transform_matrix[1,3]:.3f}, {transform_matrix[2,3]:.3f}]")
        
        # Extract 3D coordinates and RGB
        xyz = points[:, :3]
        rgb = points[:, 3:]
        
        logger.debug(f"  Pre-transform XYZ range:")
        logger.debug(f"    X: {np.min(xyz[:, 0]):.3f} to {np.max(xyz[:, 0]):.3f}m")
        logger.debug(f"    Y: {np.min(xyz[:, 1]):.3f} to {np.max(xyz[:, 1]):.3f}m")
        logger.debug(f"    Z: {np.min(xyz[:, 2]):.3f} to {np.max(xyz[:, 2]):.3f}m")
        
        # Add homogeneous coordinate
        ones = np.ones((xyz.shape[0], 1))
        xyz_homogeneous = np.hstack([xyz, ones])
        
        # Apply transformation
        xyz_transformed = (transform_matrix @ xyz_homogeneous.T).T
        xyz_transformed = xyz_transformed[:, :3]  # Remove homogeneous coordinate
        
        logger.debug(f"  Post-transform XYZ range:")
        logger.debug(f"    X: {np.min(xyz_transformed[:, 0]):.3f} to {np.max(xyz_transformed[:, 0]):.3f}m")
        logger.debug(f"    Y: {np.min(xyz_transformed[:, 1]):.3f} to {np.max(xyz_transformed[:, 1]):.3f}m")
        logger.debug(f"    Z: {np.min(xyz_transformed[:, 2]):.3f} to {np.max(xyz_transformed[:, 2]):.3f}m")
        
        # Combine with RGB
        transformed_points = np.hstack([xyz_transformed, rgb])
        logger.debug(f"  Output transformed points shape: {transformed_points.shape}")
        
        return transformed_points
    
    def crop_workspace(self, points):
        """
        Crop point cloud to workspace bounds.
        
        Args:
            points (np.ndarray): Point cloud to crop
            
        Returns:
            np.ndarray: Cropped point cloud
        """
        logger.debug(f"Cropping point cloud to workspace bounds")
        logger.debug(f"  Input points: {len(points)} points")
        
        x_min, x_max = self.workspace_bounds[0]
        y_min, y_max = self.workspace_bounds[1] 
        z_min, z_max = self.workspace_bounds[2]
        
        logger.debug(f"  Workspace bounds:")
        logger.debug(f"    X: [{x_min:.3f}, {x_max:.3f}]m")
        logger.debug(f"    Y: [{y_min:.3f}, {y_max:.3f}]m")
        logger.debug(f"    Z: [{z_min:.3f}, {z_max:.3f}]m")
        
        if len(points) == 0:
            logger.debug("  No points to crop")
            return points
        
        # Log current point cloud bounds
        logger.debug(f"  Current point cloud bounds:")
        logger.debug(f"    X: [{np.min(points[:, 0]):.3f}, {np.max(points[:, 0]):.3f}]m")
        logger.debug(f"    Y: [{np.min(points[:, 1]):.3f}, {np.max(points[:, 1]):.3f}]m")
        logger.debug(f"    Z: [{np.min(points[:, 2]):.3f}, {np.max(points[:, 2]):.3f}]m")
        
        # Create mask for points within workspace
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) & 
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        
        cropped_points = points[mask]
        points_removed = len(points) - len(cropped_points)
        removal_percentage = 100 * points_removed / len(points) if len(points) > 0 else 0
        
        logger.debug(f"  Cropping result: {len(cropped_points)} points remaining ({points_removed} removed, {removal_percentage:.1f}%)")
        
        return cropped_points
    
    def farthest_point_sampling(self, points, num_points=None):
        """
        Apply Farthest Point Sampling to reduce point cloud size.
        
        Args:
            points (np.ndarray): Input point cloud
            num_points (int): Number of points to sample
            
        Returns:
            np.ndarray: Sampled point cloud
        """
        if num_points is None:
            num_points = self.num_points
        
        logger.debug(f"Applying Farthest Point Sampling")
        logger.debug(f"  Input points: {len(points)}, target: {num_points}")
            
        if len(points) <= num_points:
            # If we have fewer points than desired, return all
            logger.debug(f"  Insufficient points for sampling, returning all {len(points)} points")
            return points
        
        # Use only XYZ for FPS
        xyz = points[:, :3]
        
        # Convert to torch tensor
        xyz_tensor = torch.from_numpy(xyz).float()
        logger.debug(f"  Created torch tensor: {xyz_tensor.shape}, device: {xyz_tensor.device}")
        
        try:
            # Apply FPS
            sampled_points, indices = torch3d_ops.sample_farthest_points(
                points=xyz_tensor.unsqueeze(0), 
                K=[num_points]
            )
            
            # Get indices as numpy
            indices = indices.squeeze(0).numpy()
            logger.debug(f"  FPS completed, sampled {len(indices)} points")
            logger.debug(f"  Index range: [{np.min(indices)}, {np.max(indices)}]")
            
            # Return sampled points (including RGB)
            sampled_result = points[indices]
            logger.debug(f"  Final sampled points shape: {sampled_result.shape}")
            
            return sampled_result
            
        except Exception as e:
            logger.error(f"  FPS failed: {e}")
            logger.warning(f"  Falling back to random sampling")
            # Fallback to random sampling
            random_indices = np.random.choice(len(points), num_points, replace=False)
            return points[random_indices]
    
    def save_frame_pcd(self, episode_dir, camera_name, frame_idx, points):
        """
        Save point cloud as PCD file in the episode directory.
        
        Args:
            episode_dir (Path): Episode directory
            camera_name (str): Name of the camera
            frame_idx (int): Frame index
            points (np.ndarray): Point cloud with shape (N, 6) [x, y, z, r, g, b]
        """
        # Create camera-specific PCD directory
        camera_folder_mapping = {
            'wrist_camera': 'wrist_pcd',
            'left_camera': 'left_pcd',
            'right_camera': 'right_pcd',
            'merged_4000': 'merged_4000',
            'merged_1024': 'merged_1024'
        }
        
        pcd_dir = episode_dir / camera_folder_mapping.get(camera_name, f"{camera_name}_pcd")
        pcd_dir.mkdir(exist_ok=True)
        
        # Create PCD file path
        pcd_file = pcd_dir / f"frame_{frame_idx:06d}.pcd"
        
        try:
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            
            # Set points (XYZ)
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            
            # Set colors (RGB) - normalize to [0, 1]
            colors = points[:, 3:6] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Save PCD file
            o3d.io.write_point_cloud(str(pcd_file), pcd)
            
            logger.debug(f"    Saved PCD: {pcd_file} ({len(points)} points)")
            
        except Exception as e:
            logger.error(f"    Failed to save PCD {pcd_file}: {e}")
    
    def process_single_camera(self, episode_dir, frame_idx, camera_name, folder_prefix, end_effector_pose=None):
        """
        Process a single camera for a specific frame.
        
        Args:
            episode_dir (Path): Episode directory
            frame_idx (int): Frame index
            camera_name (str): Camera name
            folder_prefix (str): Folder prefix for this camera
            end_effector_pose (np.ndarray): End effector pose for wrist camera (optional)
            
        Returns:
            tuple: (camera_name, points) or (camera_name, None) if failed
        """
        try:
            logger.debug(f"    Processing {camera_name} (folder: {folder_prefix})")
            
            # Try both video files and individual frames
            video_path = episode_dir / f"{folder_prefix}_camera.mp4"
            rgb_path = episode_dir / f"{folder_prefix}_images" / f"frame_{frame_idx:06d}.png"
            depth_path = episode_dir / f"{folder_prefix}_depth_images" / f"frame_{frame_idx:06d}.png"
            
            # Load RGB image (from video or individual frames)
            rgb_image = None
            if video_path.exists():
                rgb_image = self.extract_frame_from_video(video_path, frame_idx)
            elif rgb_path.exists():
                rgb_image = cv2.imread(str(rgb_path))
                if rgb_image is not None:
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            if rgb_image is None:
                logger.warning(f"    Could not load RGB image for {camera_name} frame {frame_idx}")
                return camera_name, None
            
            # Load depth image (only from individual frames)
            if not depth_path.exists():
                logger.warning(f"    Missing depth file for {camera_name} frame {frame_idx}")
                return camera_name, None
            
            depth_image = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            if depth_image is None:
                logger.warning(f"    Could not load depth image for {camera_name} frame {frame_idx}")
                return camera_name, None
            
            # Convert to point cloud
            points = self.depth_to_pointcloud(depth_image, rgb_image, camera_name)
            
            if len(points) == 0:
                logger.warning(f"    No valid points for {camera_name} frame {frame_idx}")
                return camera_name, None
            
            # Transform to base frame
            if camera_name == 'wrist_camera' and end_effector_pose is not None:
                points = self.transform_pointcloud(points, camera_name, end_effector_pose)
            else:
                points = self.transform_pointcloud(points, camera_name)
            
            return camera_name, points
            
        except Exception as e:
            logger.error(f"    Error processing {camera_name} frame {frame_idx}: {e}")
            return camera_name, None
    
    def process_frame(self, episode_dir, frame_idx, end_effector_pose=None):
        """
        Process a single frame from cameras based on camera_to_process filter.
        
        Args:
            episode_dir (Path): Episode directory
            frame_idx (int): Frame index
            end_effector_pose (np.ndarray): End effector pose for wrist camera (optional)
            
        Returns:
            dict: Dictionary containing point clouds for each camera
        """
        logger.debug(f"Processing frame {frame_idx}")
        frame_data = {}
        
        # Camera mappings (adjust based on your setup)
        all_camera_mappings = {
            'left_camera': 'left',
            'wrist_camera': 'wrist', 
            'right_camera': 'right'
        }
        
        # Filter camera mappings based on camera_to_process
        if self.camera_to_process:
            camera_name = f"{self.camera_to_process}_camera"
            if camera_name in all_camera_mappings:
                camera_mappings = {camera_name: all_camera_mappings[camera_name]}
                logger.debug(f"  Processing only {self.camera_to_process} camera")
            else:
                logger.error(f"  Invalid camera_to_process: {self.camera_to_process}")
                return frame_data
        else:
            camera_mappings = all_camera_mappings
            logger.debug(f"  Processing all cameras")
        
        logger.debug(f"  Camera mappings: {camera_mappings}")
        
        # Step 1: Process cameras in parallel and collect transformed point clouds
        camera_point_clouds = {}
        
        # Use ThreadPoolExecutor for I/O bound operations (reading images)
        with ThreadPoolExecutor(max_workers=min(3, len(camera_mappings))) as executor:
            # Submit all camera processing tasks
            future_to_camera = {}
            for camera_name, folder_prefix in camera_mappings.items():
                future = executor.submit(
                    self.process_single_camera,
                    episode_dir, frame_idx, camera_name, folder_prefix, end_effector_pose
                )
                future_to_camera[future] = camera_name
            
            # Collect results as they complete
            for future in as_completed(future_to_camera):
                camera_name, points = future.result()
                if points is not None:
                    camera_point_clouds[camera_name] = points
                    logger.debug(f"    {camera_name}: {len(points)} points processed")
        
        # Step 2: Process individual cameras if specified
        if self.camera_to_process:
            camera_name = f"{self.camera_to_process}_camera"
            if camera_name in camera_point_clouds:
                logger.debug(f"  Processing {self.camera_to_process} camera individually...")
                camera_points = camera_point_clouds[camera_name]
                
                # Crop to workspace
                logger.debug(f"    Cropping {self.camera_to_process} camera to workspace...")
                camera_points = self.crop_workspace(camera_points)
                
                if len(camera_points) > 0:
                    logger.debug(f"    After cropping: {len(camera_points)} points")
                    
                    # Apply FPS (4000 points)
                    logger.debug(f"    Applying FPS to {self.camera_to_process} camera...")
                    camera_points = self.farthest_point_sampling(camera_points, 4000)
                    
                    logger.debug(f"    Final {self.camera_to_process} point cloud: {len(camera_points)} points")
                    
                    # Save camera PCD file
                    self.save_frame_pcd(episode_dir, camera_name, frame_idx, camera_points)
                    frame_data[camera_name] = camera_points
                else:
                    logger.warning(f"    No {self.camera_to_process} points in workspace for frame {frame_idx}")
        else:
            # Step 2: Process wrist camera (individual processing, crop, FPS, save)
            if 'wrist_camera' in camera_point_clouds:
                logger.debug(f"  Processing wrist camera individually...")
                wrist_points = camera_point_clouds['wrist_camera']
                
                # Crop to workspace
                logger.debug(f"    Cropping wrist camera to workspace...")
                wrist_points = self.crop_workspace(wrist_points)
                
                if len(wrist_points) > 0:
                    logger.debug(f"    After cropping: {len(wrist_points)} points")
                    
                    # Apply FPS (4000 points)
                    logger.debug(f"    Applying FPS to wrist camera...")
                    wrist_points = self.farthest_point_sampling(wrist_points, 4000)
                    
                    logger.debug(f"    Final wrist point cloud: {len(wrist_points)} points")
                    
                    # Save wrist PCD file
                    self.save_frame_pcd(episode_dir, 'wrist_camera', frame_idx, wrist_points)
                    frame_data['wrist_camera'] = wrist_points
                else:
                    logger.warning(f"    No wrist points in workspace for frame {frame_idx}")
            
            # Step 3: Merge left and right cameras and process in parallel
            if 'left_camera' in camera_point_clouds and 'right_camera' in camera_point_clouds:
                logger.debug(f"  Merging left and right camera point clouds...")
                left_points = camera_point_clouds['left_camera']
                right_points = camera_point_clouds['right_camera']
                
                # Merge point clouds
                merged_points = np.vstack([left_points, right_points])
                logger.debug(f"    Merged point cloud: {len(merged_points)} points (left: {len(left_points)}, right: {len(right_points)})")
                
                # Crop merged point cloud to workspace
                logger.debug(f"    Cropping merged point cloud to workspace...")
                merged_points = self.crop_workspace(merged_points)
                
                if len(merged_points) > 0:
                    logger.debug(f"    After cropping: {len(merged_points)} points")
                    
                    # Process both FPS versions in parallel using ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        # Submit FPS tasks
                        future_4000 = executor.submit(self.farthest_point_sampling, merged_points, 4000)
                        future_1024 = executor.submit(self.farthest_point_sampling, merged_points, 1024)
                        
                        # Get results
                        merged_4000 = future_4000.result()
                        merged_1024 = future_1024.result()
                    
                    logger.debug(f"    Merged 4000 point cloud: {len(merged_4000)} points")
                    logger.debug(f"    Merged 1024 point cloud: {len(merged_1024)} points")
                    
                    # Save merged PCD files in parallel
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        executor.submit(self.save_frame_pcd, episode_dir, 'merged_4000', frame_idx, merged_4000)
                        executor.submit(self.save_frame_pcd, episode_dir, 'merged_1024', frame_idx, merged_1024)
                    
                    frame_data['merged_4000'] = merged_4000
                    frame_data['merged_1024'] = merged_1024
                else:
                    logger.warning(f"    No merged points in workspace for frame {frame_idx}")
            else:
                if 'left_camera' not in camera_point_clouds:
                    logger.warning(f"    Missing left camera for frame {frame_idx}")
                if 'right_camera' not in camera_point_clouds:
                    logger.warning(f"    Missing right camera for frame {frame_idx}")
        
        logger.debug(f"  Frame {frame_idx} processed successfully with {len(frame_data)} outputs")
        return frame_data
    
    def process_episode(self, episode_name):
        """
        Process a single episode.
        
        Args:
            episode_name (str): Name of the episode (e.g., 'episode_001')
        """
        episode_dir = self.dataset_path / episode_name
        
        logger.info(f"=== Processing {episode_name} ===")
        logger.info(f"Episode directory: {episode_dir}")
        
        if not episode_dir.exists():
            logger.error(f"Episode directory not found: {episode_dir}")
            return
        
        # Load episode data
        npz_path = episode_dir / "episode_data.npz"
        
        if npz_path.exists():
            logger.info(f"Loading episode data from NPZ file...")
            try:
                episode_data = np.load(npz_path)
                num_frames = len(episode_data['timestamps'])
                logger.info(f"NPZ file loaded successfully")
                logger.info(f"  Keys: {list(episode_data.keys())}")
                logger.info(f"  Number of frames: {num_frames}")
                logger.info(f"  Data shapes:")
                for key in episode_data.keys():
                    logger.info(f"    {key}: {episode_data[key].shape}")
            except Exception as e:
                logger.error(f"Failed to load NPZ file: {e}")
                episode_data = None
                num_frames = 0
        else:
            logger.info(f"No NPZ file found, counting frames from image directories or videos...")
            episode_data = None
            
            # First try to count frames from videos
            sample_video = episode_dir / "left_camera.mp4"
            if sample_video.exists():
                logger.info(f"Found video file, counting frames from: {sample_video}")
                try:
                    cap = cv2.VideoCapture(str(sample_video))
                    if cap.isOpened():
                        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        cap.release()
                        logger.info(f"Video has {num_frames} frames at {fps:.2f} FPS")
                    else:
                        logger.error(f"Could not open video file: {sample_video}")
                        num_frames = 0
                except Exception as e:
                    logger.error(f"Error reading video file: {e}")
                    num_frames = 0
            else:
                # Fallback to counting from image directories
                sample_dir = episode_dir / "left_images"
                if sample_dir.exists():
                    frame_files = list(sample_dir.glob("frame_*.png"))
                    num_frames = len(frame_files)
                    logger.info(f"Found {num_frames} frames in {sample_dir}")
                else:
                    logger.error(f"Cannot determine number of frames for {episode_name}")
                    logger.error(f"No video file or image directory found")
                    return
        
        if num_frames == 0:
            logger.error(f"No frames found for {episode_name}")
            return
        
        logger.info(f"Processing {num_frames} frames...")
        
        # Process frames in parallel batches (unless sequential mode is enabled)
        episode_point_clouds = []
        episode_robot_states = []
        failed_frames = 0
        
        if self.sequential:
            logger.info("Sequential processing mode enabled")
            # Sequential processing
            for frame_idx in tqdm(range(num_frames), desc=f"Processing {episode_name}"):
                if frame_idx % 50 == 0:
                    logger.info(f"  Processing frame {frame_idx}/{num_frames}")
                
                # Get end effector pose for this frame
                end_effector_pose = None
                if episode_data is not None:
                    try:
                        end_effector_pose = episode_data['end_effector_states'][frame_idx]
                        if frame_idx == 0:
                            logger.info(f"  Using end effector pose for wrist camera transformations")
                            logger.info(f"  Sample end effector pose (frame 0): {end_effector_pose}")
                    except Exception as e:
                        logger.warning(f"Failed to extract end effector pose for frame {frame_idx}: {e}")
                
                frame_data = self.process_frame(episode_dir, frame_idx, end_effector_pose)
                
                if frame_data:
                    episode_point_clouds.append(frame_data)
                    
                    # Store robot state
                    if episode_data is not None:
                        try:
                            robot_state = {
                                'end_effector_state': episode_data['end_effector_states'][frame_idx],
                                'joint_state': episode_data['joint_states'][frame_idx],
                                'gripper_state': episode_data['gripper_states'][frame_idx],
                                'timestamp': episode_data['timestamps'][frame_idx]
                            }
                            episode_robot_states.append(robot_state)
                        except Exception as e:
                            logger.warning(f"Failed to extract robot state for frame {frame_idx}: {e}")
                else:
                    logger.warning(f"No valid point clouds for frame {frame_idx}")
                    failed_frames += 1
        else:
            # Parallel processing
            logger.info(f"Processing frames in parallel with {self.max_workers} workers, batch size: {self.batch_size}")
            
            for batch_start in range(0, num_frames, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_frames)
                batch_frames = list(range(batch_start, batch_end))
                
                logger.info(f"  Processing batch {batch_start//self.batch_size + 1}/{(num_frames + self.batch_size - 1)//self.batch_size}: frames {batch_start}-{batch_end-1}")
                
                # Process batch of frames in parallel
                batch_results = []
                
                # Use ThreadPoolExecutor for I/O bound operations
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit frame processing tasks
                    future_to_frame = {}
                    for frame_idx in batch_frames:
                        # Get end effector pose for this frame
                        end_effector_pose = None
                        if episode_data is not None:
                            try:
                                end_effector_pose = episode_data['end_effector_states'][frame_idx]
                            except Exception as e:
                                logger.warning(f"Failed to extract end effector pose for frame {frame_idx}: {e}")
                        
                        future = executor.submit(self.process_frame, episode_dir, frame_idx, end_effector_pose)
                        future_to_frame[future] = frame_idx
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_frame):
                        frame_idx = future_to_frame[future]
                        try:
                            frame_data = future.result()
                            if frame_data:
                                batch_results.append((frame_idx, frame_data))
                            else:
                                logger.warning(f"No valid point clouds for frame {frame_idx}")
                                failed_frames += 1
                        except Exception as e:
                            logger.error(f"Error processing frame {frame_idx}: {e}")
                            failed_frames += 1
                
                # Sort batch results by frame index and add to episode data
                batch_results.sort(key=lambda x: x[0])
                for frame_idx, frame_data in batch_results:
                    episode_point_clouds.append(frame_data)
                    
                    # Store robot state if available
                    if episode_data is not None:
                        try:
                            robot_state = {
                                'end_effector_state': episode_data['end_effector_states'][frame_idx],
                                'joint_state': episode_data['joint_states'][frame_idx],
                                'gripper_state': episode_data['gripper_states'][frame_idx],
                                'timestamp': episode_data['timestamps'][frame_idx]
                            }
                            episode_robot_states.append(robot_state)
                            
                            if frame_idx == 0:  # Log first frame robot state
                                logger.info(f"  Sample robot state (frame 0):")
                                logger.info(f"    End effector: {robot_state['end_effector_state']}")
                                logger.info(f"    Gripper: {robot_state['gripper_state']}")
                                
                        except Exception as e:
                            logger.warning(f"Failed to extract robot state for frame {frame_idx}: {e}")
                
                # Progress update
                processed_frames = len(episode_point_clouds)
                progress = 100 * processed_frames / num_frames
                logger.info(f"  Batch completed. Total processed: {processed_frames}/{num_frames} ({progress:.1f}%)")
        
        # Processing summary
        successful_frames = len(episode_point_clouds)
        success_rate = 100 * successful_frames / num_frames if num_frames > 0 else 0
        
        logger.info(f"Episode processing summary:")
        logger.info(f"  Total frames: {num_frames}")
        logger.info(f"  Successfully processed: {successful_frames}")
        logger.info(f"  Failed frames: {failed_frames}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        
        if successful_frames == 0:
            logger.error(f"No frames were successfully processed for {episode_name}")
            return
        
        # Analyze point cloud statistics
        total_points = 0
        camera_stats = {}
        for frame_data in episode_point_clouds:
            for output_name, points in frame_data.items():
                if output_name not in camera_stats:
                    camera_stats[output_name] = []
                camera_stats[output_name].append(len(points))
                total_points += len(points)
        
        logger.info(f"Point cloud statistics:")
        logger.info(f"  Total points across all frames: {total_points}")
        for output_name, point_counts in camera_stats.items():
            avg_points = np.mean(point_counts)
            min_points = np.min(point_counts)
            max_points = np.max(point_counts)
            logger.info(f"  {output_name}: avg={avg_points:.1f}, min={min_points}, max={max_points} points per frame")
        
        # Save episode summary (much smaller than full data)
        summary_file = episode_dir / "processing_summary.json"
        
        summary_data = {
            'processing_metadata': {
                'num_frames_processed': len(episode_point_clouds),
                'original_num_frames': num_frames,
                'failed_frames': failed_frames,
                'success_rate': success_rate,
                'cameras': list(self.extrinsics.keys()),
                'outputs': list(camera_stats.keys()),
                'workspace_bounds': self.workspace_bounds,
                'num_points_per_cloud': self.num_points,
                'output_statistics': {k: {
                    'avg_points': float(np.mean(v)),
                    'min_points': int(np.min(v)),
                    'max_points': int(np.max(v)),
                    'frames_processed': len(v)
                } for k, v in camera_stats.items()}
            }
        }
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            logger.info(f"Saved processing summary: {summary_file}")
        except Exception as e:
            logger.warning(f"Failed to save processing summary: {e}")
        
        logger.info(f"PCD files saved in output-specific directories within: {episode_dir}")
        if self.camera_to_process:
            logger.info(f"  - {self.camera_to_process}_pcd/ ({self.camera_to_process} camera point clouds - 4000 points)")
        else:
            logger.info(f"  - wrist_pcd/ (wrist camera point clouds - 4000 points)")
            logger.info(f"  - merged_4000/ (merged left+right point clouds - 4000 points)")
            logger.info(f"  - merged_1024/ (merged left+right point clouds - 1024 points)")
        logger.info(f"=== Finished processing {episode_name} ===\n")
    
    def process_all_episodes(self, episode_filter=None):
        """
        Process all episodes in the dataset.
        
        Args:
            episode_filter (list): List of episode names to process (if None, process all)
        """
        logger.info("=== Starting Dataset Processing ===")
        logger.info(f"Dataset path: {self.dataset_path}")
        logger.info(f"Workspace bounds: {self.workspace_bounds}")
        logger.info(f"Points per cloud (FPS): {self.num_points}")
        if self.camera_to_process:
            logger.info(f"Processing only {self.camera_to_process} camera")
            logger.info(f"PCD files will be saved in: {self.camera_to_process}_pcd/ (4000 points)")
        else:
            logger.info("Processing all cameras")
            logger.info("PCD files will be saved directly in each episode directory")
            logger.info("  - wrist_pcd/ (4000 points)")
            logger.info("  - merged_4000/ (left+right merged, 4000 points)")
            logger.info("  - merged_1024/ (left+right merged, 1024 points)")
        
        # Get all episode directories
        logger.info("Scanning for episode directories...")
        episode_dirs = [d for d in self.dataset_path.iterdir() 
                       if d.is_dir() and d.name.startswith('episode_')]
        
        logger.info(f"Found {len(episode_dirs)} total episodes")
        
        if episode_filter:
            logger.info(f"Filtering episodes: {episode_filter}")
            episode_dirs = [d for d in episode_dirs if d.name in episode_filter]
            logger.info(f"Episodes after filtering: {len(episode_dirs)}")
        
        episode_dirs.sort()
        
        if len(episode_dirs) == 0:
            logger.error("No episodes found to process!")
            return
        
        
        # Process episodes in parallel or sequentially
        successful_episodes = 0
        failed_episodes = []
        
        if self.sequential or len(episode_dirs) == 1:
            # Sequential processing
            logger.info("Using sequential episode processing")
            for i, episode_dir in enumerate(episode_dirs):
                logger.info(f"\n>>> Processing episode {i+1}/{len(episode_dirs)}: {episode_dir.name}")
                
                try:
                    self.process_episode(episode_dir.name)
                    successful_episodes += 1
                    logger.info(f"✓ Successfully processed {episode_dir.name}")
                    
                except Exception as e:
                    logger.error(f"✗ Error processing {episode_dir.name}: {e}")
                    failed_episodes.append(episode_dir.name)
                    continue
        else:
            # Parallel processing for multiple episodes
            max_episode_workers = min(mp.cpu_count() // 2, 4)  # Limit to avoid resource contention
            logger.info(f"Processing episodes in parallel with {max_episode_workers} workers")
            
            with ProcessPoolExecutor(max_workers=max_episode_workers) as executor:
                # Create a partial function with the converter's configuration
                process_episode_func = partial(
                    process_episode_worker,
                    dataset_path=self.dataset_path,
                    cameras_path=self.cameras_path,
                    workspace_bounds=self.workspace_bounds,
                    num_points=self.num_points,
                    camera_to_process=self.camera_to_process
                )
                
                # Submit episode processing tasks
                future_to_episode = {}
                for episode_dir in episode_dirs:
                    future = executor.submit(process_episode_func, episode_dir.name)
                    future_to_episode[future] = episode_dir.name
                
                # Collect results with progress tracking
                with tqdm(total=len(episode_dirs), desc="Processing episodes") as pbar:
                    for future in as_completed(future_to_episode):
                        episode_name = future_to_episode[future]
                        try:
                            success = future.result()
                            if success:
                                successful_episodes += 1
                                logger.info(f"✓ Successfully processed {episode_name}")
                            else:
                                failed_episodes.append(episode_name)
                                logger.error(f"✗ Failed to process {episode_name}")
                        except Exception as e:
                            logger.error(f"✗ Error processing {episode_name}: {e}")
                            failed_episodes.append(episode_name)
                        
                        pbar.update(1)
        
        # Final summary
        logger.info("\n=== Processing Complete ===")
        logger.info(f"Total episodes processed: {len(episode_dirs)}")
        logger.info(f"Successful: {successful_episodes}")
        logger.info(f"Failed: {len(failed_episodes)}")
        
        if failed_episodes:
            logger.warning(f"Failed episodes: {failed_episodes}")
        
        if successful_episodes > 0:
            logger.info(f"PCD files saved in episode directories under: {self.dataset_path}")
            if self.camera_to_process:
                logger.info(f"Each episode contains: {self.camera_to_process}_pcd/ (4000 points)")
            else:
                logger.info("Each episode contains output-specific PCD directories:")
                logger.info("  - wrist_pcd/ (wrist camera - 4000 points)")
                logger.info("  - merged_4000/ (left+right merged - 4000 points)")
                logger.info("  - merged_1024/ (left+right merged - 1024 points)")
        
        logger.info("=== Dataset Processing Finished ===")
        return successful_episodes, failed_episodes

def main():
    parser = argparse.ArgumentParser(description='Convert franka teleop dataset to point clouds')
    parser.add_argument('--dataset_path', type=str, default='/mnt/SharedDrive/franka_recordings',
                       help='Path to the franka recordings dataset')
    parser.add_argument('--cameras_path', type=str, default='/home/jeeveshm/franka_teleop/cameras.json',
                       help='Path to cameras.json file containing intrinsics and extrinsics')
    parser.add_argument('--output_path', type=str, default='/home/jeeveshm/franka_teleop/processed_data',
                       help='Output directory for processed point clouds')
    parser.add_argument('--num_points', type=int, default=4000,
                       help='Number of points for FPS sampling')
    parser.add_argument('--episodes', nargs='+', default=None,
                       help='Specific episodes to process (e.g., episode_001 episode_002)')
    parser.add_argument('--workspace_bounds', nargs=6, type=float, default=None,
                       help='Workspace bounds as x_min x_max y_min y_max z_min z_max')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose debug logging')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: auto-detect)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for frame processing (default: auto-detect)')
    parser.add_argument('--sequential', action='store_true',
                       help='Force sequential processing (disable parallelization)')
    parser.add_argument('--camera_to_process', type=str, choices=['left', 'right', 'wrist'], default=None,
                       help='Process only specific camera (left, right, or wrist). If not specified, processes all cameras.')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    
    # Parse workspace bounds
    workspace_bounds = None
    if args.workspace_bounds:
        workspace_bounds = [
            [args.workspace_bounds[0], args.workspace_bounds[1]],
            [args.workspace_bounds[2], args.workspace_bounds[3]],
            [args.workspace_bounds[4], args.workspace_bounds[5]]
        ]
        logger.info(f"Parsed workspace bounds: {workspace_bounds}")
    
    try:
        # Initialize converter
        logger.info("Initializing converter...")
        converter = DatasetConverter(
            dataset_path=args.dataset_path,
            cameras_path=args.cameras_path,
            output_path=args.output_path,
            num_points=args.num_points,
            workspace_bounds=workspace_bounds,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            sequential=args.sequential,
            camera_to_process=args.camera_to_process
        )
        
        # Process episodes
        logger.info("Starting episode processing...")
        successful, failed = converter.process_all_episodes(episode_filter=args.episodes)
        
        # Final status
        if successful > 0:
            logger.info(f"✓ Processing completed successfully! {successful} episodes processed.")
        else:
            logger.error("✗ No episodes were processed successfully!")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
