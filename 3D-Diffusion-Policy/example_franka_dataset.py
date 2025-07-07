#!/usr/bin/env python3

"""
Example usage of FrankaDataset for custom episode format.

This script demonstrates how to:
1. Load custom episode data with RGBD images
2. Convert depth images to point clouds on-the-fly
3. Use the dataset with the same interface as AdroitDataset
"""

import numpy as np
import torch
from diffusion_policy_3d.dataset.franka_dataset import FrankaDataset

def main():
    # Configuration for your custom Franka dataset
    dataset_config = {
        'dataset_path': '/path/to/your/franka/dataset',  # Update this path
        'horizon': 16,
        'pad_before': 1,
        'pad_after': 7,
        'seed': 42,
        'val_ratio': 0.1,
        'max_train_episodes': None,  # Use all episodes
        'task_name': 'franka_custom',
        'use_point_cloud': True,
        'num_points': 512,
        'point_cloud_sampling_method': 'fps',
        'camera_intrinsics': {
            'fx': 525.0,
            'fy': 525.0,
            'cx': 319.5,
            'cy': 239.5,
            'width': 640,
            'height': 480
        },
        'workspace_bounds': [
            [0.3, 0.8],   # x bounds
            [-0.3, 0.3],  # y bounds
            [0.0, 0.5]    # z bounds
        ]
    }
    
    # Create dataset
    print("Creating FrankaDataset...")
    dataset = FrankaDataset(**dataset_config)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Create validation dataset
    val_dataset = dataset.get_validation_dataset()
    print(f"Validation dataset created with {len(val_dataset)} samples")
    
    # Get normalizer
    normalizer = dataset.get_normalizer()
    print("Normalizer created")
    
    # Example usage: iterate through dataset
    print("\nExample data loading:")
    for i, batch in enumerate(dataset):
        if i >= 3:  # Only show first 3 samples
            break
            
        print(f"\nSample {i}:")
        print(f"  Point cloud shape: {batch['obs']['point_cloud'].shape}")
        print(f"  Agent pos shape: {batch['obs']['agent_pos'].shape}")
        print(f"  Action shape: {batch['action'].shape}")
        
        # Print some statistics
        pc = batch['obs']['point_cloud']
        print(f"  Point cloud range: [{pc.min():.3f}, {pc.max():.3f}]")
        
        agent_pos = batch['obs']['agent_pos']
        print(f"  Agent pos range: [{agent_pos.min():.3f}, {agent_pos.max():.3f}]")
        
        action = batch['action']
        print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
    
    # Example: use with DataLoader
    print("\nExample with DataLoader:")
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    for i, batch in enumerate(dataloader):
        if i >= 2:  # Only show first 2 batches
            break
            
        print(f"\nBatch {i}:")
        print(f"  Point cloud shape: {batch['obs']['point_cloud'].shape}")
        print(f"  Agent pos shape: {batch['obs']['agent_pos'].shape}")
        print(f"  Action shape: {batch['action'].shape}")

if __name__ == "__main__":
    main()
