class ObservationHistoryManager:
    def __init__(self, max_history_length=2):
        self.max_history_length = max_history_length
        self.observation_history = []

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
        required_length = self.max_history_length
        current_length = len(self.observation_history)
        if current_length < required_length:
            # Repeat the first observation to pad
            padding_needed = required_length - current_length
            first_obs = self.observation_history[0]
            padded_history = [first_obs] * padding_needed + self.observation_history
        else:
            padded_history = self.observation_history[-required_length:]
        point_clouds = []
        agent_poses = []
        for obs in padded_history:
            point_clouds.append(obs['point_cloud'])
            agent_poses.append(obs['agent_pos'])
        point_cloud_array = np.stack(point_clouds)  # [T, N, 3] or [T, 1, 1024, 3]
        agent_pos_array = np.stack(agent_poses)     # [T, 1, 1, 8] or [T, 1, 8]
        # Remove extra singleton dimensions if present
        point_cloud_array = np.squeeze(point_cloud_array)
        agent_pos_array = np.squeeze(agent_pos_array)
        # Add batch dimension
        point_cloud_array = np.expand_dims(point_cloud_array, axis=0)  # [1, T, ...]
        agent_pos_array = np.expand_dims(agent_pos_array, axis=0)      # [1, T, ...]
        obs_dict = {
            'point_cloud': point_cloud_array,
            'agent_pos': agent_pos_array
        }
        return obs_dict
#!/usr/bin/env python3

import socket
import pickle
import threading
import torch
import numpy as np
import argparse
import os
import sys
import boto3
import hydra
from omegaconf import OmegaConf
from termcolor import cprint
# Add the project path
sys.path.append('/mnt/data/Diffusion3d/3D-Diffusion-Policy')

# Import the diffusion policy components
from train_modified import TrainDP3Workspace

class DiffusionPolicyServer:
    def __init__(self, config_path, s3_bucket="pr-checkpoints", latest=True, epoch=None, 
                 device="cuda", use_ema=False, restore_checkpoint=True):
        self.device = device
        self.use_ema = use_ema
        
        # Load the model
        self.load_model(config_path, s3_bucket, latest, epoch, restore_checkpoint)
        print("Diffusion Policy Server initialized and ready for inference")
        
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
            
            # Create workspace with model
            self.workspace = TrainDP3Workspace(self.cfg)
            
            # Load checkpoint from S3 or local cache
            local_path = checkpoint_name
            
            # if s3_bucket and s3_bucket.lower() != "null":
            #     cprint(f"Using S3 bucket: {s3_bucket}")
            #     s3_client = boto3.client('s3')
            #     if latest:
            #         s3_path = "DP3_outputs/latest/checkpoint.pth"
            #     else:
            #         s3_path = f"DP3_outputs/epoch_{epoch}/checkpoint.pth"
                
            #     if restore_checkpoint and os.path.exists(local_path):
            #         print(f"Loading checkpoint from local cache: {local_path}")
            #     else:
            #         print(f"Downloading checkpoint from S3: {s3_bucket}/{s3_path}")
            #         s3_client.download_file(s3_bucket, s3_path, local_path)
            # else:
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
            
    def run_inference(self, obs_dict):
        """Run inference with the diffusion policy model"""
        try:
            # Convert numpy arrays to torch tensors if needed
            if isinstance(obs_dict['point_cloud'], np.ndarray):
                obs_dict['point_cloud'] = torch.from_numpy(obs_dict['point_cloud']).float().to(self.device)
            if isinstance(obs_dict['agent_pos'], np.ndarray):
                obs_dict['agent_pos'] = torch.from_numpy(obs_dict['agent_pos']).float().to(self.device)
            
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

def handle_client(conn, server_instance):
    try:
        # Receive data from client
        data = b""
        while True:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet
            if data.endswith(b"<END>"):
                data = data[:-5]
                break
        
        # Deserialize observation
        obs_dict = pickle.loads(data)
        print(f"Received observation with shapes: point_cloud={obs_dict['point_cloud'].shape}, agent_pos={obs_dict['agent_pos'].shape}")
        
        # Run inference
        actions = server_instance.run_inference(obs_dict)
        
        if actions is not None:
            print(f"Generated actions with shape: {actions.shape}")
            response = pickle.dumps(actions)
        else:
            print("Inference failed, sending None")
            response = pickle.dumps(None)
            
        # Send response back to client
        conn.sendall(response + b"<END>")
        
    except Exception as e:
        print(f"Error handling client: {e}")
        # Send error response
        try:
            response = pickle.dumps(None)
            conn.sendall(response + b"<END>")
        except:
            pass
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="3D Diffusion Policy Inference Server")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the config file used for training")
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
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host address")
    parser.add_argument("--port", type=int, default=5000,
                        help="Server port")
    
    args = parser.parse_args()
    
    # # Initialize the server
    server_instance = DiffusionPolicyServer(
        config_path=args.config_path,
        s3_bucket=args.s3_bucket,
        latest=args.latest,
        epoch=args.epoch,
        device=args.device,
        use_ema=args.use_ema,
        restore_checkpoint=args.restore_checkpoint
    )
    
    # Start the socket server
    # server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # server.bind((args.host, args.port))
    # server.listen(5)
    # print(f"Server listening on {args.host}:{args.port}")
    
    # try:
    #     while True:
    #         conn, addr = server.accept()
    #         print(f"Connection from {addr}")
    #         threading.Thread(target=handle_client, args=(conn, server_instance)).start()
    # except KeyboardInterrupt:
    #     print("Shutting down server...")
    # finally:
    #     server.close()

    episode_path = "/home/jeeveshm/franka_teleop/test_recordings/episode_001/"
    point_cloud_folder = os.path.join(episode_path, "merged_1024/")
    episode_data = os.path.join(episode_path, "episode_data.npz")
    point_cloud_files = sorted(os.listdir(point_cloud_folder))
    num_frames = len(point_cloud_files)
    print(f"Number of frames in episode: {num_frames}")
    point_clouds = []
    for i in range(num_frames):
        point_cloud_path = os.path.join(point_cloud_folder, point_cloud_files[i])
        import open3d as o3d
        point_cloud = o3d.io.read_point_cloud(point_cloud_path)
        point_cloud = np.asarray(point_cloud.points)
        #pcd shape is (1024, 3) make it (1,1,1024,3)
        point_cloud = point_cloud[np.newaxis, np.newaxis, :, :]  # Reshape to (1, 1, 1024, 3)
        point_clouds.append(point_cloud)
    print(f"Point clouds shape: {len(point_clouds)} frames, each with shape {point_clouds[0].shape}")
    joints = np.load(episode_data)['joint_states']
    gripper = np.load(episode_data)['gripper_states']
    agent_poses = np.concatenate((joints, gripper), axis=-1)  # Concatenate along the last dimension
    agent_poses = agent_poses[np.newaxis, np.newaxis, :, :]
    #reshape from (1,1,num_frames,8) to (num_frames, 1, 1, 8)
    agent_poses = agent_poses.reshape(num_frames, 1, 1, -1)  # Reshape to (num_frames, 1, 1, 8)
    #make agent_poses also a list of num_frames with each having shape (1, 1, 8)
    agent_poses = [agent_poses[i] for i in range(num_frames)]
    print(f"Agent poses shape: {len(agent_poses)} frames, each with shape {agent_poses[0].shape}")
    #Similar;y get actions from same numpy file
    gello_jnts = np.load(episode_data)['gello_joint_states']
    gello_gripper = np.load(episode_data)['gello_gripper_percent'].reshape(-1, 1)  # Ensure gripper is reshaped to (num_frames, 1
    actions_gt = np.concatenate((gello_jnts, gello_gripper), axis=-1)  # Concatenate along the last dimension
    actions_gt = actions_gt[np.newaxis, np.newaxis, :, :]  # Reshape to (1, 1, num_frames, 8)
    actions_gt = actions_gt.reshape(num_frames, 1, 1, -1)  # Reshape to (num_frames, 1, 1, 8)
    actions_gt = [actions_gt[i] for i in range(num_frames)]
    # Use observation history manager for stacking
    max_history_length = 2  # Set as needed for your model
    import socket
    obs_hist_mgr = ObservationHistoryManager(max_history_length=max_history_length)
    host = '127.0.0.1'  # Use localhost for client connection
    port = 5000
    import time
    all_actions = []
    for i in range(num_frames):
        obs = {
            'point_cloud': point_clouds[i],
            'agent_pos': agent_poses[i]
        }
        obs_hist_mgr.update_observation_history(obs)
        model_input = obs_hist_mgr.create_model_input()
        if model_input is None:
            print(f"Skipping frame {i+1}: not enough history.")
            all_actions.append(np.zeros((1, 8), dtype=np.float32))
            continue
        #print model input 
        print(f"Model input for frame {i+1}: point_cloud shape {model_input['point_cloud'].shape}, agent_pos shape {model_input['agent_pos'].shape}")
        #print agent_pos
        print(f"Agent position for frame {i+1}: {model_input['agent_pos']}")
        actions = server_instance.run_inference(model_input)
        if actions is not None:
            print(f"Actions for frame {i+1}: {actions[0,0,:]}")
            gt_action = actions_gt[i]
            print(f"Ground truth action for frame {i+1}: {gt_action}")
            all_actions.append(actions[0,0,:].reshape(1, 8))
        else:
            print(f"Inference failed for frame {i+1}")
            all_actions.append(np.zeros((1, 8), dtype=np.float32))
        time.sleep(0.01)

    # Stack all actions into [num_frames, 8]
    all_actions_array = np.concatenate(all_actions, axis=0)  # shape: [num_frames, 8]
    print(f"All actions shape: {all_actions_array.shape}")  
    # save actions to a file
    output_file = os.path.join(episode_path, "inferred_actions_infer.npy")
    np.save(output_file, all_actions_array)
    print(f"All actions saved to {output_file}")    
   
if __name__ == "__main__":
    main()

