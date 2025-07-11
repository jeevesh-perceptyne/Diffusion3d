if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import time
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
import boto3
from botocore.exceptions import NoCredentialsError

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self.s3_bucket = 'pr-checkpoints'
        self.s3_client = boto3.client('s3')
        cprint(OmegaConf.to_yaml(cfg), 'cyan')
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:
                self.ema_model = hydra.utils.instantiate(cfg.policy)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0
        self.lr_scheduler = None  # will be set in run()
        print(f"Initial global step: {self.global_step}, epoch: {self.epoch}")

    def s3_ckpt_path(self, epoch=None, latest=False):
        if latest:
            return f"DP3_outputs/latest/checkpoint.pth"
        else:
            return f"DP3_outputs/epoch_{epoch}/checkpoint.pth"

    def save_checkpoint_s3(self, epoch, latest=False):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'config': OmegaConf.to_container(self.cfg, resolve=True),
        }
        # Save normalizer state_dict for robust inference
        if hasattr(self.model, 'normalizer') and self.model.normalizer is not None:
            checkpoint['normalizer_state_dict'] = self.model.normalizer.state_dict()
        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        
        local_path = f'checkpoint_tmp.pth'
        torch.save(checkpoint, local_path)
        
        if latest:
            # Save only to latest path when latest=True
            s3_path = self.s3_ckpt_path(latest=True)
        else:
            # Save only to epoch-specific path when latest=False
            s3_path = self.s3_ckpt_path(epoch=epoch)
        
        self.s3_client.upload_file(local_path, self.s3_bucket, s3_path)
        
        # Also save config as yaml for easy inspection/inference
        config_yaml_path = 'config_tmp.yaml'
        with open(config_yaml_path, 'w') as f:
            OmegaConf.save(self.cfg, f)
        s3_config_path = s3_path.replace('checkpoint.pth', 'config.yaml')
        self.s3_client.upload_file(config_yaml_path, self.s3_bucket, s3_config_path)
        
        os.remove(local_path)
        os.remove(config_yaml_path)
        print(f"Checkpoint and config saved to S3: {s3_path}")

    def load_latest_checkpoint_s3(self, device):
        s3_latest = self.s3_ckpt_path(latest=True)
        local_path = 'checkpoint_latest_tmp.pth'
        try:
            self.s3_client.download_file(self.s3_bucket, s3_latest, local_path)
            ckpt = torch.load(local_path, map_location=device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if self.lr_scheduler and 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict']:
                self.lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if self.ema_model and 'ema_state_dict' in ckpt:
                self.ema_model.load_state_dict(ckpt['ema_state_dict'])
            # Restore normalizer state_dict if present
            if hasattr(self.model, 'normalizer') and 'normalizer_state_dict' in ckpt:
                self.model.normalizer.load_state_dict(ckpt['normalizer_state_dict'])
            self.epoch = ckpt['epoch']
            self.global_step = ckpt['global_step']
            # Optionally restore config
            if 'config' in ckpt:
                self.cfg = OmegaConf.create(ckpt['config'])
            cprint(f"Resumed from checkpoint at epoch {self.epoch}, step {self.global_step}", 'green')
            os.remove(local_path)
        except Exception as e:
            print(f"No checkpoint found or error loading checkpoint: {e}")

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = False
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = False
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost
        
        # Resume from latest S3 checkpoint if requested
        device = torch.device(cfg.training.device)
        if cfg.training.resume:
            self.load_latest_checkpoint_s3(device)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        print(f"Dataset: {dataset}")

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
    
        # configure lr scheduler
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )
        
        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
        env_runner = None
        # configure env
        # env_runner: BaseRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)

        # if env_runner is not None:
        #     assert isinstance(env_runner, BaseRunner)
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()  
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': self.lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                t3 = time.time()
                # runner_log = env_runner.run(policy, dataset=dataset)
                runner_log = env_runner.run(policy)
                t4 = time.time()
                # print(f"rollout time: {t4-t3:.3f}")
                # log all
                step_log.update(runner_log)

            
                
            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    
                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            if env_runner is None:
                step_log['test_mean_score'] = - train_loss
                
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            
            # S3 Checkpoint saving strategy:
            # 1. Save latest checkpoint after every epoch (overwrites previous latest)
            self.save_checkpoint_s3(epoch=self.epoch, latest=True)
            
            # 2. Save numbered checkpoint every 500 epochs
            if (self.epoch + 1) % 500 == 0:
                self.save_checkpoint_s3(epoch=self.epoch + 1, latest=False)
            self.epoch += 1
            del step_log

    def eval(self):
        """
        Evaluation function - loads latest S3 checkpoint and runs evaluation
        """
        cfg = copy.deepcopy(self.cfg)
        device = torch.device(cfg.training.device)
        
        # Load latest checkpoint from S3
        self.load_latest_checkpoint_s3(device)
        
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)
        
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    import sys
    
    # Parse command line arguments for number of episodes
    max_episodes = None
    pcd_type = 'merged_1024'  # default
    
    # Simple argument parsing
    for i, arg in enumerate(sys.argv):
        if arg == '--max_episodes' and i + 1 < len(sys.argv):
            try:
                max_episodes = int(sys.argv[i + 1])
                print(f"Using max_episodes: {max_episodes}")
            except ValueError:
                print(f"Invalid max_episodes value: {sys.argv[i + 1]}")
        elif arg == '--pcd_type' and i + 1 < len(sys.argv):
            pcd_type = sys.argv[i + 1]
            print(f"Using pcd_type: {pcd_type}")
    
    # Override dataset configuration
    if max_episodes is not None:
        cfg.task.dataset.max_train_episodes = max_episodes
    
    # Set PCD type
    cfg.task.dataset.pcd_type = pcd_type
    
    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    print(f"Running from {pathlib.Path(__file__).parent}")
    main()
