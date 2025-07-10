#!/usr/bin/env python3
"""
Test script to verify the FrankaDataset works with PCD files
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add the project root to Python path
sys.path.append('/home/jeeveshm/Diffusion3d/3D-Diffusion-Policy')

from diffusion_policy_3d.dataset.franka_dataset import FrankaDataset

def test_dataset():
    """Test the FrankaDataset with PCD files"""
    
    # Dataset configuration
    dataset_config = {
        'dataset_path': '/mnt/SharedDrive/franka_recordings',
        'horizon': 16,
        'max_train_episodes': 3,  # Test with first 3 episodes
        'use_point_cloud': True,
        'num_points': 1024,
        'pcd_type': 'merged_1024',  # Use merged_1024 PCD files
        'val_ratio': 0.2
    }
    
    print("Initializing FrankaDataset...")
    try:
        dataset = FrankaDataset(**dataset_config)
        print(f"✓ Dataset initialized successfully")
        print(f"  Total samples: {len(dataset)}")
        
        # Test loading a single sample
        print("\nTesting single sample loading...")
        sample = dataset[0]
        print(f"✓ Sample loaded successfully")
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Point cloud shape: {sample['obs']['point_cloud'].shape}")
        print(f"  Agent pos shape: {sample['obs']['agent_pos'].shape}")
        print(f"  Action shape: {sample['action'].shape}")
        
        # Test DataLoader
        print("\nTesting DataLoader...")
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        
        for i, batch in enumerate(dataloader):
            print(f"✓ Batch {i} loaded successfully")
            print(f"  Point cloud batch shape: {batch['obs']['point_cloud'].shape}")
            print(f"  Agent pos batch shape: {batch['obs']['agent_pos'].shape}")
            print(f"  Action batch shape: {batch['action'].shape}")
            
            if i >= 2:  # Test first 3 batches
                break
        
        # Test validation dataset
        print("\nTesting validation dataset...")
        val_dataset = dataset.get_validation_dataset()
        print(f"✓ Validation dataset created")
        print(f"  Validation samples: {len(val_dataset)}")
        
        # Test normalizer
        print("\nTesting normalizer...")
        normalizer = dataset.get_normalizer()
        print(f"✓ Normalizer created successfully")
        
        print("\n🎉 All tests passed! Dataset is ready for training.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_dataset()
    sys.exit(0 if success else 1)
