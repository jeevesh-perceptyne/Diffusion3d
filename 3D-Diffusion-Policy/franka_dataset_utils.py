#!/usr/bin/env python3

"""
Utility script for FrankaDataset preprocessing and validation.

This script provides utilities to:
1. Validate dataset format
2. Visualize point clouds
3. Check data statistics
4. Convert existing data to the expected format
"""

import os
import numpy as np
import cv2
import argparse
from typing import List, Dict, Optional
from pathlib import Path


def validate_dataset_format(dataset_path: str) -> bool:
    """
    Validate that the dataset follows the expected format.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        True if format is valid, False otherwise
    """
    print(f"Validating dataset format at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return False
    
    # Find episode directories
    episode_dirs = [d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('episode_')]
    
    if len(episode_dirs) == 0:
        print("ERROR: No episode directories found (should start with 'episode_')")
        return False
    
    print(f"Found {len(episode_dirs)} episode directories")
    
    valid_episodes = 0
    for episode_dir in episode_dirs:
        episode_path = os.path.join(dataset_path, episode_dir)
        
        # Check for episode_data.npz
        episode_data_path = os.path.join(episode_path, 'episode_data.npz')
        if not os.path.exists(episode_data_path):
            print(f"WARNING: Missing episode_data.npz in {episode_dir}")
            continue
        
        # Check episode data content
        try:
            episode_data = np.load(episode_data_path, allow_pickle=True)
            required_keys = ['states', 'actions']
            alt_keys = ['agent_pos', 'action']
            
            has_required = any(key in episode_data for key in required_keys)
            has_alt = any(key in episode_data for key in alt_keys)
            
            if not (has_required or has_alt):
                print(f"WARNING: Episode {episode_dir} missing required keys. "
                      f"Expected: {required_keys} or {alt_keys}")
                continue
        except Exception as e:
            print(f"WARNING: Could not load episode_data.npz in {episode_dir}: {e}")
            continue
        
        # Check for depth image directories
        depth_dirs = ['left_depth_images', 'right_depth_images', 'wrist_depth_images']
        found_depth_dirs = []
        
        for depth_dir in depth_dirs:
            depth_path = os.path.join(episode_path, depth_dir)
            if os.path.exists(depth_path):
                found_depth_dirs.append(depth_dir)
        
        if len(found_depth_dirs) == 0:
            print(f"WARNING: No depth image directories found in {episode_dir}")
        else:
            print(f"  {episode_dir}: Found depth dirs: {found_depth_dirs}")
        
        valid_episodes += 1
    
    print(f"\nValidation complete: {valid_episodes}/{len(episode_dirs)} episodes are valid")
    return valid_episodes > 0


def check_dataset_statistics(dataset_path: str) -> Dict:
    """
    Check dataset statistics.
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        Dictionary with statistics
    """
    print(f"Checking dataset statistics at: {dataset_path}")
    
    stats = {
        'total_episodes': 0,
        'total_steps': 0,
        'episode_lengths': [],
        'state_dims': [],
        'action_dims': [],
        'depth_image_counts': [],
    }
    
    episode_dirs = [d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('episode_')]
    
    for episode_dir in episode_dirs:
        episode_path = os.path.join(dataset_path, episode_dir)
        episode_data_path = os.path.join(episode_path, 'episode_data.npz')
        
        if not os.path.exists(episode_data_path):
            continue
        
        try:
            episode_data = np.load(episode_data_path, allow_pickle=True)
            
            # Get states and actions
            states = episode_data.get('states', episode_data.get('agent_pos', None))
            actions = episode_data.get('actions', episode_data.get('action', None))
            
            if states is not None and actions is not None:
                stats['total_episodes'] += 1
                episode_length = len(states)
                stats['total_steps'] += episode_length
                stats['episode_lengths'].append(episode_length)
                stats['state_dims'].append(states.shape[-1] if states.ndim > 1 else 1)
                stats['action_dims'].append(actions.shape[-1] if actions.ndim > 1 else 1)
                
                # Count depth images
                depth_count = 0
                for depth_dir in ['left_depth_images', 'right_depth_images', 'wrist_depth_images']:
                    depth_path = os.path.join(episode_path, depth_dir)
                    if os.path.exists(depth_path):
                        depth_images = [f for f in os.listdir(depth_path) if f.endswith('.png')]
                        depth_count += len(depth_images)
                
                stats['depth_image_counts'].append(depth_count)
        
        except Exception as e:
            print(f"Error processing {episode_dir}: {e}")
    
    # Calculate statistics
    if stats['episode_lengths']:
        stats['avg_episode_length'] = np.mean(stats['episode_lengths'])
        stats['min_episode_length'] = np.min(stats['episode_lengths'])
        stats['max_episode_length'] = np.max(stats['episode_lengths'])
    
    if stats['state_dims']:
        stats['state_dim'] = stats['state_dims'][0]  # Assuming consistent
    
    if stats['action_dims']:
        stats['action_dim'] = stats['action_dims'][0]  # Assuming consistent
    
    if stats['depth_image_counts']:
        stats['avg_depth_images'] = np.mean(stats['depth_image_counts'])
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Average episode length: {stats.get('avg_episode_length', 0):.1f}")
    print(f"  Episode length range: [{stats.get('min_episode_length', 0)}, {stats.get('max_episode_length', 0)}]")
    print(f"  State dimension: {stats.get('state_dim', 'Unknown')}")
    print(f"  Action dimension: {stats.get('action_dim', 'Unknown')}")
    print(f"  Average depth images per episode: {stats.get('avg_depth_images', 0):.1f}")
    
    return stats


def visualize_sample_data(dataset_path: str, episode_idx: int = 0, step_idx: int = 0):
    """
    Visualize sample data from the dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        episode_idx: Index of episode to visualize
        step_idx: Index of step within episode to visualize
    """
    print(f"Visualizing sample data: episode {episode_idx}, step {step_idx}")
    
    episode_dirs = sorted([d for d in os.listdir(dataset_path) 
                          if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('episode_')])
    
    if episode_idx >= len(episode_dirs):
        print(f"ERROR: Episode index {episode_idx} out of range (0-{len(episode_dirs)-1})")
        return
    
    episode_dir = episode_dirs[episode_idx]
    episode_path = os.path.join(dataset_path, episode_dir)
    
    # Load episode data
    episode_data_path = os.path.join(episode_path, 'episode_data.npz')
    episode_data = np.load(episode_data_path, allow_pickle=True)
    
    states = episode_data.get('states', episode_data.get('agent_pos', None))
    actions = episode_data.get('actions', episode_data.get('action', None))
    
    if step_idx >= len(states):
        print(f"ERROR: Step index {step_idx} out of range (0-{len(states)-1})")
        return
    
    print(f"Episode {episode_dir}, Step {step_idx}:")
    print(f"  State: {states[step_idx]}")
    print(f"  Action: {actions[step_idx]}")
    
    # Check depth images
    for depth_dir in ['left_depth_images', 'right_depth_images', 'wrist_depth_images']:
        depth_path = os.path.join(episode_path, depth_dir)
        if os.path.exists(depth_path):
            depth_img_path = os.path.join(depth_path, f'frame_{step_idx:06d}.png')
            if os.path.exists(depth_img_path):
                depth_img = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
                if depth_img is not None:
                    print(f"  {depth_dir}: {depth_img.shape}, range: [{depth_img.min()}, {depth_img.max()}]")
                else:
                    print(f"  {depth_dir}: Could not load image")
            else:
                print(f"  {depth_dir}: Image not found")
        else:
            print(f"  {depth_dir}: Directory not found")


def create_sample_episode(output_path: str, episode_length: int = 10):
    """
    Create a sample episode for testing purposes.
    
    Args:
        output_path: Path to create the sample episode
        episode_length: Length of the episode to create
    """
    print(f"Creating sample episode at: {output_path}")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Create sample episode data
    states = np.random.randn(episode_length, 7)  # 7-DOF robot state
    actions = np.random.randn(episode_length, 7)  # 7-DOF actions
    
    np.savez(
        os.path.join(output_path, 'episode_data.npz'),
        states=states,
        actions=actions
    )
    
    # Create sample depth images
    for camera in ['left', 'right', 'wrist']:
        depth_dir = os.path.join(output_path, f'{camera}_depth_images')
        os.makedirs(depth_dir, exist_ok=True)
        
        for i in range(episode_length):
            # Create random depth image
            depth_img = np.random.randint(0, 65535, (480, 640), dtype=np.uint16)
            cv2.imwrite(os.path.join(depth_dir, f'frame_{i:06d}.png'), depth_img)
    
    print(f"Sample episode created with {episode_length} steps")


def main():
    parser = argparse.ArgumentParser(description='FrankaDataset utilities')
    parser.add_argument('command', choices=['validate', 'stats', 'visualize', 'create_sample'],
                       help='Command to execute')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to the dataset directory')
    parser.add_argument('--episode_idx', type=int, default=0,
                       help='Episode index for visualization')
    parser.add_argument('--step_idx', type=int, default=0,
                       help='Step index for visualization')
    parser.add_argument('--episode_length', type=int, default=10,
                       help='Length of sample episode to create')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        validate_dataset_format(args.dataset_path)
    elif args.command == 'stats':
        check_dataset_statistics(args.dataset_path)
    elif args.command == 'visualize':
        visualize_sample_data(args.dataset_path, args.episode_idx, args.step_idx)
    elif args.command == 'create_sample':
        create_sample_episode(args.dataset_path, args.episode_length)


if __name__ == "__main__":
    main()
