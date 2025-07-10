import wandb
import numpy as np
import torch
import tqdm
from typing import Dict
from termcolor import cprint

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util


class FrankaRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 dataset_path="/mnt/SharedDrive/franka_recordings",
                 eval_episodes=20,
                 max_steps=400,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 camera_names=None,
                 use_point_cloud=True,
                 num_points=512,
                 horizon=16,
                 ):
        super().__init__(output_dir)
        
        self.dataset_path = dataset_path
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.crf = crf
        self.tqdm_interval_sec = tqdm_interval_sec
        self.camera_names = camera_names or ['left_camera', 'right_camera', 'wrist_camera']
        self.use_point_cloud = use_point_cloud
        self.num_points = num_points
        self.horizon = horizon
        
        # Initialize loggers
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        
        # Load validation dataset for evaluation
        self._load_validation_data()
    
    def _load_validation_data(self):
        """Load validation episodes for evaluation"""
        from diffusion_policy_3d.dataset.franka_dataset import FrankaDataset
        
        dataset_config = {
            'dataset_path': self.dataset_path,
            'horizon': self.horizon,
            'pad_before': 1,
            'pad_after': 7,
            'seed': 42,
            'val_ratio': 0.1,
            'max_train_episodes': None,
            'task_name': 'franka_custom',
            'use_point_cloud': self.use_point_cloud,
            'num_points': self.num_points,
            'point_cloud_sampling_method': 'fps',
            'camera_names': self.camera_names,
            'camera_intrinsics': {
                'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5,
                'width': 640, 'height': 480
            },
            'workspace_bounds': [
                [0.3, 0.8], [-0.3, 0.3], [0.0, 0.5]
            ]
        }
        
        # Create dataset and get validation subset
        full_dataset = FrankaDataset(**dataset_config)
        self.val_dataset = full_dataset.get_validation_dataset()
        
        print(f"FrankaRunner initialized with {len(self.val_dataset)} validation samples")
    
    def run(self, policy: BasePolicy, save_video=False) -> Dict:
        """
        Run evaluation on validation dataset
        Since we don't have a real environment, we'll evaluate on held-out validation data
        """
        device = policy.device
        dtype = policy.dtype
        
        # Metrics tracking
        all_action_errors = []
        all_prediction_accuracies = []
        all_trajectories = []
        
        # Limit evaluation episodes to available data
        eval_episodes = min(self.eval_episodes, len(self.val_dataset))
        
        for episode_idx in tqdm.tqdm(range(eval_episodes), 
                                   desc="Evaluating on Franka validation data", 
                                   leave=False, 
                                   mininterval=self.tqdm_interval_sec):
            
            # Get a sample from validation dataset
            sample = self.val_dataset[episode_idx]
            
            # Extract observation and ground truth action
            obs = sample['obs']
            gt_action = sample['action']  # Ground truth action sequence
            
            # Prepare observation for policy
            obs_dict = dict_apply(obs, lambda x: x.to(device=device).unsqueeze(0))
            
            # Get policy prediction
            policy.reset()
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            
            # Extract predicted action
            pred_action = action_dict['action'].squeeze(0)  # Remove batch dimension
            
            # Convert to numpy for evaluation
            gt_action_np = gt_action.cpu().numpy()
            pred_action_np = pred_action.cpu().numpy()
            
            # Compute metrics
            # 1. Action L2 error
            action_error = np.mean(np.linalg.norm(gt_action_np - pred_action_np, axis=-1))
            all_action_errors.append(action_error)
            
            # 2. Trajectory accuracy (how close the predicted trajectory is to ground truth)
            trajectory_accuracy = self._compute_trajectory_accuracy(gt_action_np, pred_action_np)
            all_prediction_accuracies.append(trajectory_accuracy)
            
            # Store trajectory for visualization
            all_trajectories.append({
                'gt_action': gt_action_np,
                'pred_action': pred_action_np,
                'obs': obs
            })
        
        # Compute final metrics
        mean_action_error = np.mean(all_action_errors)
        mean_prediction_accuracy = np.mean(all_prediction_accuracies)
        
        # Log data
        log_data = dict()
        log_data['mean_action_error'] = mean_action_error
        log_data['mean_prediction_accuracy'] = mean_prediction_accuracy
        log_data['test_mean_score'] = mean_prediction_accuracy  # Use accuracy as main score
        
        # Update loggers
        self.logger_util_test.record(mean_prediction_accuracy)
        self.logger_util_test10.record(mean_prediction_accuracy)
        log_data['accuracy_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['accuracy_test_L5'] = self.logger_util_test10.average_of_largest_K()
        
        # Print results
        cprint(f"Mean Action Error: {mean_action_error:.4f}", 'yellow')
        cprint(f"Mean Prediction Accuracy: {mean_prediction_accuracy:.4f}", 'green')
        cprint(f"Test Mean Score: {mean_prediction_accuracy:.4f}", 'green')
        
        # Create visualization if requested
        if save_video and len(all_trajectories) > 0:
            self._create_trajectory_visualization(all_trajectories, log_data)
        
        return log_data
    
    def _compute_trajectory_accuracy(self, gt_action, pred_action):
        """
        Compute trajectory accuracy based on action similarity
        Returns a score between 0 and 1
        """
        # Compute normalized error
        error = np.linalg.norm(gt_action - pred_action, axis=-1)
        max_error = np.linalg.norm(gt_action, axis=-1) + 1e-6  # Avoid division by zero
        
        # Compute accuracy (1 - normalized error)
        accuracy = 1.0 - np.mean(error / max_error)
        return max(0.0, accuracy)  # Clamp to [0, 1]
    
    def _create_trajectory_visualization(self, trajectories, log_data):
        """
        Create trajectory visualization for wandb
        """
        try:
            import matplotlib.pyplot as plt
            
            # Select a few trajectories for visualization
            n_viz = min(3, len(trajectories))
            
            fig, axes = plt.subplots(n_viz, 2, figsize=(12, 4 * n_viz))
            if n_viz == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(n_viz):
                traj = trajectories[i]
                gt_action = traj['gt_action']
                pred_action = traj['pred_action']
                
                # Plot joint positions
                ax1 = axes[i, 0]
                for j in range(min(7, gt_action.shape[-1])):  # Plot first 7 joints
                    ax1.plot(gt_action[:, j], label=f'GT Joint {j}', linestyle='--')
                    ax1.plot(pred_action[:, j], label=f'Pred Joint {j}', linestyle='-')
                ax1.set_title(f'Trajectory {i+1} - Joint Positions')
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Joint Position')
                ax1.legend()
                
                # Plot error over time
                ax2 = axes[i, 1]
                error = np.linalg.norm(gt_action - pred_action, axis=-1)
                ax2.plot(error, color='red', label='L2 Error')
                ax2.set_title(f'Trajectory {i+1} - Prediction Error')
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('L2 Error')
                ax2.legend()
            
            plt.tight_layout()
            
            # Save to wandb
            log_data['trajectory_comparison'] = wandb.Image(fig)
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not create trajectory visualization: {e}")
    
    def run_real_robot_evaluation(self, policy: BasePolicy):
        """
        Placeholder for real robot evaluation
        This would be implemented when you have a real robot setup
        """
        cprint("Real robot evaluation not implemented yet", 'yellow')
        cprint("This would require:", 'yellow')
        cprint("1. Robot control interface", 'yellow')
        cprint("2. Camera feed processing", 'yellow')
        cprint("3. Safety systems", 'yellow')
        cprint("4. Task success metrics", 'yellow')
        
        return {
            'real_robot_score': 0.0,
            'message': 'Real robot evaluation not implemented'
        }