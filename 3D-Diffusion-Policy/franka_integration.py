#!/usr/bin/env python3

"""
Integration script showing how to use FrankaDataset with the 3D Diffusion Policy training pipeline.

This script demonstrates:
1. Loading FrankaDataset in the training pipeline
2. Using it with existing training configurations
3. Adapting the config for custom dataset
"""

import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

def create_franka_config(dataset_path: str, 
                        state_dim: int = 7, 
                        action_dim: int = 7, 
                        num_points: int = 512) -> DictConfig:
    """
    Create a configuration for FrankaDataset training.
    
    Args:
        dataset_path: Path to your Franka dataset
        state_dim: Dimension of robot state
        action_dim: Dimension of robot action
        num_points: Number of points in point cloud
        
    Returns:
        Configuration dictionary
    """
    
    config = OmegaConf.create({
        # Task configuration
        'name': 'franka_custom',
        'task_name': 'franka_custom',
        
        # Training parameters
        'horizon': 16,
        'n_obs_steps': 2,
        'n_action_steps': 8,
        'n_latency_steps': 0,
        'dataset_obs_steps': 2,
        'past_action_visible': False,
        'keypoint_visible_rate': 1.0,
        'n_envs': 1,
        
        # Shape metadata
        'shape_meta': {
            'obs': {
                'point_cloud': {
                    'shape': [num_points, 3],
                    'type': 'point_cloud'
                },
                'agent_pos': {
                    'shape': [state_dim],
                    'type': 'low_dim'
                }
            },
            'action': {
                'shape': [action_dim]
            }
        },
        
        # Dataset configuration
        'dataset': {
            '_target_': 'diffusion_policy_3d.dataset.franka_dataset.FrankaDataset',
            'dataset_path': dataset_path,
            'horizon': '${horizon}',
            'pad_before': '${eval:${n_obs_steps}-1}',
            'pad_after': '${eval:${n_action_steps}-1}',
            'seed': 42,
            'val_ratio': 0.02,
            'max_train_episodes': None,
            'task_name': 'franka_custom',
            'use_point_cloud': True,
            'num_points': num_points,
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
        },
        
        # Policy configuration (DP3)
        'policy': {
            '_target_': 'diffusion_policy_3d.policy.dp3.DP3Policy',
            'shape_meta': '${shape_meta}',
            'noise_scheduler': {
                '_target_': 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler',
                'num_train_timesteps': 100,
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'beta_schedule': 'squaredcos_cap_v2',
                'variance_type': 'fixed_small',
                'clip_sample': True,
                'prediction_type': 'epsilon'
            },
            'horizon': '${horizon}',
            'n_action_steps': '${n_action_steps}',
            'n_obs_steps': '${n_obs_steps}',
            'num_inference_steps': 100,
            'obs_encoder': {
                '_target_': 'diffusion_policy_3d.model.vision.pointnet_extractor.DP3Encoder',
                'observation_space': '${shape_meta.obs}',
                'pointcloud_encoder_cfg': {
                    'in_channels': 3,
                    'out_channels': 256,
                    'use_layernorm': True,
                    'final_norm': 'layernorm'
                },
                'use_pc_color': False,
                'pointnet_type': 'pointnet'
            },
            'use_point_crop': True
        },
        
        # Dataloader configuration
        'dataloader': {
            'batch_size': 64,
            'num_workers': 4,
            'shuffle': True,
            'pin_memory': True,
            'persistent_workers': True
        },
        
        # Optimizer configuration
        'optimizer': {
            '_target_': 'torch.optim.AdamW',
            'lr': 1e-4,
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'weight_decay': 1e-6
        },
        
        # Training configuration
        'training': {
            'device': 'cuda:0',
            'seed': 42,
            'resume': True,
            'lr_scheduler': 'cosine',
            'lr_warmup_steps': 500,
            'num_epochs': 3000,
            'gradient_accumulate_every': 1,
            'max_train_steps': None,
            'max_val_steps': None,
            'rollout_every': 50,
            'checkpoint_every': 50,
            'val_every': 1,
            'sample_every': 5,
            'tqdm_interval_sec': 1.0,
            'freeze_encoder': False
        },
        
        # Logging configuration
        'logging': {
            'project': 'franka_dp3',
            'resume': True,
            'mode': 'online',
            'name': '${now:%Y.%m.%d-%H.%M.%S}_${name}',
            'tags': ['${name}', '${task_name}'],
            'id': None,
            'group': None
        },
        
        # Checkpoint configuration
        'checkpoint': {
            'topk': {
                'monitor': 'val_loss',
                'mode': 'min',
                'k': 5,
                'format_str': 'epoch={epoch:04d}-val_loss={val_loss:.3f}.ckpt'
            },
            'save_last_ckpt': True,
            'save_last_snapshot': False
        },
        
        # Multirun configuration
        'multi_run': {
            'run_dir': 'data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}',
            'wandb_name_base': '${now:%Y.%m.%d-%H.%M.%S}_${name}'
        }
    })
    
    return config


def test_franka_dataset_loading(config: DictConfig):
    """
    Test loading the FrankaDataset with the given configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("Testing FrankaDataset loading...")
    
    try:
        # Import here to avoid issues if not installed
        from diffusion_policy_3d.dataset.franka_dataset import FrankaDataset
        
        # Create dataset
        dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
        dataset = FrankaDataset(**dataset_config)
        
        print(f"✓ Dataset loaded successfully with {len(dataset)} samples")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"✓ Sample shape - Point cloud: {sample['obs']['point_cloud'].shape}, "
              f"Agent pos: {sample['obs']['agent_pos'].shape}, "
              f"Action: {sample['action'].shape}")
        
        # Test validation dataset
        val_dataset = dataset.get_validation_dataset()
        print(f"✓ Validation dataset created with {len(val_dataset)} samples")
        
        # Test normalizer
        normalizer = dataset.get_normalizer()
        print("✓ Normalizer created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False


def save_config_file(config: DictConfig, output_path: str):
    """
    Save the configuration to a YAML file.
    
    Args:
        config: Configuration to save
        output_path: Path to save the configuration
    """
    with open(output_path, 'w') as f:
        OmegaConf.save(config, f)
    print(f"Configuration saved to: {output_path}")


def main():
    """
    Main function to demonstrate FrankaDataset integration.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='FrankaDataset integration script')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to your Franka dataset')
    parser.add_argument('--state_dim', type=int, default=7,
                       help='Dimension of robot state')
    parser.add_argument('--action_dim', type=int, default=7,
                       help='Dimension of robot action')
    parser.add_argument('--num_points', type=int, default=512,
                       help='Number of points in point cloud')
    parser.add_argument('--output_config', type=str, default='franka_config.yaml',
                       help='Output configuration file path')
    parser.add_argument('--test_loading', action='store_true',
                       help='Test loading the dataset')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_franka_config(
        dataset_path=args.dataset_path,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        num_points=args.num_points
    )
    
    # Save configuration
    save_config_file(config, args.output_config)
    
    # Test loading if requested
    if args.test_loading:
        success = test_franka_dataset_loading(config)
        if success:
            print("\n✓ All tests passed! Your FrankaDataset is ready to use.")
        else:
            print("\n✗ Tests failed. Please check your dataset format and configuration.")
    
    print(f"\nTo use this configuration with the training script:")
    print(f"python train.py --config-path . --config-name {args.output_config}")


if __name__ == "__main__":
    main()
