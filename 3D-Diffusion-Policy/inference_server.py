#!/usr/bin/env python3

import socket
import pickle
import threading
import torch
import numpy as np
import argparse
import os
import sys
import traceback
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
            normalizer_loaded = False
            if hasattr(self.workspace.model, 'normalizer') and 'normalizer_state_dict' in ckpt:
                self.workspace.model.normalizer.load_state_dict(ckpt['normalizer_state_dict'])
                normalizer_loaded = True
                print("Loaded normalizer from checkpoint")
            elif hasattr(self.workspace, 'normalizer') and 'normalizer_state_dict' in ckpt:
                self.workspace.normalizer.load_state_dict(ckpt['normalizer_state_dict'])
                normalizer_loaded = True
                print("Loaded normalizer from workspace")
            else:
                print("Warning: No normalizer found in checkpoint - inference may be incorrect!")
            
            # Store normalizer for easy access
            if normalizer_loaded:
                if hasattr(self.workspace.model, 'normalizer'):
                    self.normalizer = self.workspace.model.normalizer
                elif hasattr(self.workspace, 'normalizer'):
                    self.normalizer = self.workspace.normalizer
                else:
                    self.normalizer = None
            else:
                self.normalizer = None
            
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
        """Run inference with the diffusion policy model - MATCH TRAINING EXACTLY"""
        try:
            # Debug: Print input shapes
            print(f"Input shapes - point_cloud: {obs_dict['point_cloud'].shape}, agent_pos: {obs_dict['agent_pos'].shape}")
            
            # Convert numpy arrays to torch tensors if needed
            if isinstance(obs_dict['point_cloud'], np.ndarray):
                obs_dict['point_cloud'] = torch.from_numpy(obs_dict['point_cloud']).float().to(self.device)
            if isinstance(obs_dict['agent_pos'], np.ndarray):
                obs_dict['agent_pos'] = torch.from_numpy(obs_dict['agent_pos']).float().to(self.device)
            
            print(f"Converted to tensors - point_cloud: {obs_dict['point_cloud'].shape}, agent_pos: {obs_dict['agent_pos'].shape}")
            
            # Apply normalization exactly like during training
            obs_dict_normalized = dict()
            if hasattr(self.workspace.model, 'normalizer') and self.workspace.model.normalizer is not None:
                print("Applying normalizer from model")
                # Normalize using the model's normalizer (loaded from checkpoint)
                normalizer = self.workspace.model.normalizer
                obs_dict_normalized['agent_pos'] = normalizer['agent_pos'].normalize(obs_dict['agent_pos'])
                obs_dict_normalized['point_cloud'] = normalizer['point_cloud'].normalize(obs_dict['point_cloud'])
            elif hasattr(self, 'normalizer') and self.normalizer is not None:
                print("Applying normalizer from server")
                # Use server normalizer if available
                obs_dict_normalized['agent_pos'] = self.normalizer['agent_pos'].normalize(obs_dict['agent_pos'])
                obs_dict_normalized['point_cloud'] = self.normalizer['point_cloud'].normalize(obs_dict['point_cloud'])
            else:
                print("WARNING: No normalizer found, using raw observations")
                obs_dict_normalized = obs_dict
            
            # Create the obs dict in the format expected by the model
            formatted_obs = {
                'obs': obs_dict_normalized
            }
            
            # Select the appropriate model
            if self.use_ema and self.workspace.ema_model is not None:
                policy = self.workspace.ema_model
                print("Using EMA model for inference")
            else:
                policy = self.workspace.model
                print("Using main model for inference")
                
            # Run inference
            with torch.no_grad():
                result = policy.predict_action(formatted_obs)
                pred_action = result['action_pred']  # [batch, horizon, action_dim]
                
            print(f"Action prediction shape: {pred_action.shape}")
            return pred_action.cpu().numpy()
            
        except Exception as e:
            print(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
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
        
        # Validate input shapes - more flexible validation
        batch_size, time_steps = obs_dict['point_cloud'].shape[:2]
        num_points = obs_dict['point_cloud'].shape[2] if len(obs_dict['point_cloud'].shape) > 2 else obs_dict['point_cloud'].shape[1]
        
        print(f"Input validation: batch={batch_size}, time_steps={time_steps}, points={num_points}")
        
        if obs_dict['point_cloud'].shape[-1] != 3:
            raise ValueError(f"Point cloud should have 3 coordinates (x,y,z), got {obs_dict['point_cloud'].shape[-1]}")
        
        if num_points != 1024:
            print(f"Warning: Expected 1024 points, got {num_points}")
            
        if obs_dict['agent_pos'].shape[-1] != 8:
            raise ValueError(f"Agent position should have 8 dimensions, got {obs_dict['agent_pos'].shape[-1]}")
        
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
        import traceback
        traceback.print_exc()
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
    
    # Initialize the server
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
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(5)
    print(f"Server listening on {args.host}:{args.port}")
    
    try:
        while True:
            conn, addr = server.accept()
            print(f"Connection from {addr}")
            threading.Thread(target=handle_client, args=(conn, server_instance)).start()
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        server.close()

if __name__ == "__main__":
    main()

