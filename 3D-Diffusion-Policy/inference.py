import argparse
import torch
import numpy as np
import os
import sys
import copy
import boto3
from omegaconf import OmegaConf
from termcolor import cprint
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import hydra
from hydra.utils import instantiate

# Import from our project
from train_modified import TrainDP3Workspace
from diffusion_policy_3d.dataset.franka_dataset import FrankaDataset


def parse_args():
    """Parse command line arguments for the inference script."""
    parser = argparse.ArgumentParser(description="3D Diffusion Policy Inference")
    
    # Checkpoint arguments
    parser.add_argument("--s3_bucket", type=str, default="pr-checkpoints",
                        help="S3 bucket containing the checkpoint")
    parser.add_argument("--latest", action="store_true", default=True,
                        help="Use the latest checkpoint")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific epoch checkpoint to load")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the config file used for training")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to dataset (defaults to config value)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    
    # Model arguments
    parser.add_argument("--use_ema", action="store_true",
                        help="Use EMA model for inference")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--restore_checkpoint", action="store_true",
                        help="Restore from a local checkpoint if it exists, skipping S3 download.")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Directory to save results")
    parser.add_argument("--visualize", action="store_true",
                        help="Create visualizations of predicted vs actual actions")
    
    return parser.parse_args()


def get_s3_checkpoint_path(latest=True, epoch=None):
    """Get the S3 path to the checkpoint."""
    if latest:
        return "DP3_outputs/latest/checkpoint.pth"
    elif epoch is not None:
        return f"DP3_outputs/epoch_{epoch}/checkpoint.pth"
    else:
        raise ValueError("Either latest must be True or epoch must be specified")


def load_model(config_path, s3_bucket, s3_path, device="cuda", restore_checkpoint=False):
    """Load the model from S3 checkpoint."""
    # Load config using hydra's compose API to handle defaults
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)
    config_name = os.path.basename(config_path)
    
    hydra.initialize(config_path=os.path.relpath(config_dir, start=os.getcwd()), version_base=None)
    cfg = hydra.compose(config_name=config_name)
    
    # Create workspace with model
    workspace = TrainDP3Workspace(cfg)
    
    # Load checkpoint from S3 or local cache
    s3_client = boto3.client('s3')
    local_path = 'checkpoint_latest.pth' # Use a persistent name

    if restore_checkpoint and os.path.exists(local_path):
        cprint(f"Restoring from local checkpoint: {local_path}", "yellow")
    else:
        cprint(f"Downloading checkpoint from s3://{s3_bucket}/{s3_path}", "cyan")
        s3_client.download_file(s3_bucket, s3_path, local_path)
    
    try:
        # Set weights_only=False as our checkpoint contains non-tensor data (e.g. config)
        ckpt = torch.load(local_path, map_location=device, weights_only=False)
        workspace.model.load_state_dict(ckpt['model_state_dict'])
        
        # Load EMA model if available
        if workspace.ema_model is not None and 'ema_state_dict' in ckpt:
            workspace.ema_model.load_state_dict(ckpt['ema_state_dict'])
        
        # Load normalizer state (essential for proper inference)
        if hasattr(workspace.model, 'normalizer') and 'normalizer_state_dict' in ckpt:
            workspace.model.normalizer.load_state_dict(ckpt['normalizer_state_dict'])
        
        cprint(f"Loaded checkpoint from epoch {ckpt['epoch']}", "green")
        
    except Exception as e:
        cprint(f"Error loading checkpoint: {e}", "red")
        raise
    
    # Move models to device
    workspace.model.to(device)
    if workspace.ema_model is not None:
        workspace.ema_model.to(device)
    
    return workspace, cfg


def load_dataset(cfg, dataset_path=None, num_samples=None):
    """Load a dataset for testing."""
    # Make a copy of the config to avoid modifying the original
    dataset_cfg = copy.deepcopy(cfg.task.dataset)
    
    # Override dataset path if provided
    if dataset_path is not None:
        dataset_cfg.dataset_path = dataset_path
    
    # Create dataset instance
    dataset = FrankaDataset(
        dataset_path=dataset_cfg.dataset_path,
        horizon=dataset_cfg.horizon,
        pad_before=dataset_cfg.pad_before,
        pad_after=dataset_cfg.pad_after,
        seed=dataset_cfg.seed,
        val_ratio=dataset_cfg.val_ratio,
        max_train_episodes=dataset_cfg.max_train_episodes,
        task_name=dataset_cfg.task_name,
        use_point_cloud=dataset_cfg.use_point_cloud,
        num_points=dataset_cfg.num_points,
        point_cloud_sampling_method=dataset_cfg.point_cloud_sampling_method,
        pcd_type=dataset_cfg.pcd_type
    )
    
    # Get validation dataset (to test on unseen data)
    val_dataset = dataset.get_validation_dataset()
    
    # Limit number of samples if specified
    if num_samples is not None and num_samples < len(val_dataset):
        val_dataset.sample_indices = val_dataset.sample_indices[:num_samples]
        
    cprint(f"Loaded test dataset with {len(val_dataset)} samples", "green")
    
    return val_dataset


def predict_actions(model, dataset, device, use_ema=False, batch_size=4):
    """Run inference on dataset and return predictions."""
    # Select the appropriate model
    if isinstance(model, TrainDP3Workspace):
        policy = model.ema_model if use_ema and model.ema_model is not None else model.model
    else:
        policy = model
    
    policy.eval()  # Set to evaluation mode
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Collect results
    results = {
        'predictions': [],
        'ground_truth': [],
        'observations': []
    }
    
    # Run inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            # Move batch to device
            batch = {
                k: (
                    {k2: v2.to(device) for k2, v2 in v.items()} 
                    if isinstance(v, dict) 
                    else v.to(device)
                ) 
                for k, v in batch.items()
            }
            
            # Get observations and ground truth
            obs_dict = batch['obs']
            gt_action = batch['action']
            
            # Predict actions (normalization happens inside the model)
            result = policy.predict_action(obs_dict)
            pred_action = result['action_pred']
            
            # Store results
            results['predictions'].append(pred_action.cpu())
            results['ground_truth'].append(gt_action.cpu())
            results['observations'].append({
                k: v.cpu() if isinstance(v, torch.Tensor) else v 
                for k, v in obs_dict.items()
            })
    
    # Concatenate results
    results['predictions'] = torch.cat(results['predictions'], dim=0)
    results['ground_truth'] = torch.cat(results['ground_truth'], dim=0)
    
    return results


def evaluate_predictions(results):
    """Compute metrics comparing predicted and ground truth actions."""
    predictions = results['predictions']
    ground_truth = results['ground_truth']
    
    # Overall metrics
    mse = torch.nn.functional.mse_loss(predictions, ground_truth).item()
    mae = torch.nn.functional.l1_loss(predictions, ground_truth).item()
    
    # Per-dimension metrics
    per_dim_mse = torch.nn.functional.mse_loss(
        predictions, ground_truth, reduction='none'
    ).mean(dim=(0, 1)).tolist()  # Average over batch and time
    
    # Per-trajectory metrics
    traj_mse = torch.nn.functional.mse_loss(
        predictions, ground_truth, reduction='none'
    ).mean(dim=(1, 2)).tolist()  # Average over time and dimensions
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'per_dim_mse': per_dim_mse,
        'traj_mse': traj_mse
    }
    
    return metrics


def visualize_results(results, metrics, output_dir):
    """Create visualizations of predicted vs actual actions."""
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = results['predictions']
    ground_truth = results['ground_truth']
    
    # Define more intuitive names for each action dimension
    action_names = [f"Joint {i+1}" for i in range(7)] + ["Gripper"]
    
    # Select a few samples to visualize
    n_samples = min(5, len(predictions))
    
    for i in range(n_samples):
        pred = predictions[i].numpy()  # [T, action_dim]
        gt = ground_truth[i].numpy()   # [T, action_dim]
        
        n_dims = pred.shape[1]
        # Create a 2x4 grid of subplots. sharey=False allows each plot to have its own y-axis scale,
        # which is useful when actions have different ranges.
        fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=False)
        axes = axes.flatten() # Flatten the 2D array of axes for easy iteration
        
        for dim in range(n_dims):
            ax = axes[dim]
            ax.plot(gt[:, dim], 'b-', label='Ground Truth', linewidth=2, alpha=0.8)
            ax.plot(pred[:, dim], 'r--', label='Prediction', linewidth=2, alpha=0.8)
            ax.set_title(action_names[dim])
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Add a legend only to the first subplot to avoid clutter
            if dim == 0:
                ax.legend()
                
        # Set common labels
        for ax in axes[-4:]: # Set x-label for the bottom row
            ax.set_xlabel('Time Steps')

        fig.suptitle(f'Sample {i+1} Trajectory Comparison (Overall MSE: {metrics["traj_mse"][i]:.5f})', fontsize=16)
        # Adjust layout to prevent titles from overlapping
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'sample_{i+1}_comparison.png'))
        plt.close(fig)
    
    # Plot per-dimension error (this part is fine as is)
    fig, ax = plt.subplots(figsize=(12, 6))
    dims = np.arange(len(metrics['per_dim_mse']))
    ax.bar(dims, metrics['per_dim_mse'], color='skyblue', edgecolor='black')
    ax.set_xticks(dims)
    ax.set_xticklabels(action_names, rotation=45, ha="right")
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title('Prediction Error by Action Dimension', fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dimension_error.png'))
    plt.close(fig)
    
    cprint(f"Visualizations saved to {output_dir}", "green")


def run_single_inference(model, point_cloud, agent_pos, device="cuda", use_ema=False):
    """
    Run inference on a single observation - useful for streaming data.
    
    Args:
        model: The model or workspace
        point_cloud: Point cloud tensor [n_obs_steps, N, 3]
        agent_pos: Agent position tensor [n_obs_steps, 8]
        device: Device to run on
        use_ema: Whether to use EMA model
        
    Returns:
        Predicted action sequence
    """
    if isinstance(model, TrainDP3Workspace):
        policy = model.ema_model if use_ema and model.ema_model is not None else model.model
    else:
        policy = model
    
    policy.eval()
    
    # Add a batch dimension to the observation sequence
    point_cloud = point_cloud.unsqueeze(0) # [1, n_obs_steps, N, 3]
    agent_pos = agent_pos.unsqueeze(0)     # [1, n_obs_steps, 8]
    
    # Create observation dict
    obs_dict = {
        'point_cloud': point_cloud.to(device),
        'agent_pos': agent_pos.to(device)
    }
    
    # Run inference
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
        pred_action = result['action_pred']
    
    return pred_action.cpu()


def main():
    """Main entry point for the inference script."""
    # Parse command line arguments
    args = parse_args()
    
    # Get S3 checkpoint path
    s3_path = get_s3_checkpoint_path(latest=args.latest, epoch=args.epoch)
    
    # Load model from checkpoint
    workspace, cfg = load_model(
        args.config_path, args.s3_bucket, s3_path, args.device,
        restore_checkpoint=args.restore_checkpoint
    )
    
    # Load test dataset
    test_dataset = load_dataset(
        cfg, args.dataset_path, args.num_samples
    )
    
    # Run inference
    results = predict_actions(
        model=workspace,
        dataset=test_dataset,
        device=args.device,
        use_ema=args.use_ema,
        batch_size=args.batch_size
    )
    
    # Evaluate predictions
    metrics = evaluate_predictions(results)
    
    # Print evaluation results
    cprint("======== Evaluation Results ========", "magenta")
    cprint(f"Overall MSE: {metrics['mse']:.6f}", "magenta")
    cprint(f"Overall MAE: {metrics['mae']:.6f}", "magenta")
    
    cprint("\nPer-dimension MSE:", "magenta")
    for i, mse in enumerate(metrics['per_dim_mse']):
        cprint(f"  Dimension {i}: {mse:.6f}", "magenta")
    
    traj_mses = metrics['traj_mse']
    cprint(f"\nTrajectory MSE - Min: {min(traj_mses):.6f}, Max: {max(traj_mses):.6f}, Avg: {sum(traj_mses)/len(traj_mses):.6f}", "magenta")
    
    # Create visualizations if requested
    if args.visualize:
        visualize_results(results, metrics, args.output_dir)
    
    # Demonstrate single observation inference (for streaming use case)
    cprint("\nDemonstrating single observation inference:", "cyan")
    sample = test_dataset[0]
    
    # Extract an observation sequence with n_obs_steps
    # The model was trained on a sequence, not a single frame.
    n_obs_steps = cfg.n_obs_steps
    point_cloud_seq = sample['obs']['point_cloud'][:n_obs_steps]
    agent_pos_seq = sample['obs']['agent_pos'][:n_obs_steps]
    
    # Run single inference
    pred = run_single_inference(
        model=workspace,
        point_cloud=point_cloud_seq,
        agent_pos=agent_pos_seq,
        device=args.device,
        use_ema=args.use_ema
    )
    
    cprint(f"Single observation prediction shape: {pred.shape}", "cyan")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())