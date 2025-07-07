# FrankaDataset - Custom Episode Format Support

This document explains how to use the `FrankaDataset` class to work with custom episode data format where episodes contain RGBD images that are converted to point clouds on-the-fly.

## Dataset Format

Your custom dataset should follow this structure:

```
your_dataset/
├── episode_001/
│   ├── episode_data.npz
│   ├── left_camera.mp4
│   ├── right_camera.mp4
│   ├── left_depth_images/
│   │   ├── frame_000000.png
│   │   ├── frame_000001.png
│   │   └── ...
│   ├── right_depth_images/
│   │   ├── frame_000000.png
│   │   ├── frame_000001.png
│   │   └── ...
│   └── wrist_depth_images/
│       ├── frame_000000.png
│       ├── frame_000001.png
│       └── ...
├── episode_002/
│   └── ...
└── episode_XXX/
    └── ...
```

### Episode Data NPZ Format

Each `episode_data.npz` file should contain:
- `states` or `agent_pos`: Robot state data (numpy array)
- `actions` or `action`: Action data (numpy array)
- Any other relevant data

### Depth Images

- Depth images should be saved as PNG files
- Named as `frame_XXXXXX.png` where XXXXXX is the 6-digit frame number
- Can have multiple camera views: `left_depth_images/`, `right_depth_images/`, `wrist_depth_images/`

## Usage

### Basic Usage

```python
from diffusion_policy_3d.dataset.franka_dataset import FrankaDataset

# Create dataset
dataset = FrankaDataset(
    dataset_path='/path/to/your/franka/dataset',
    horizon=16,
    pad_before=1,
    pad_after=7,
    seed=42,
    val_ratio=0.1,
    max_train_episodes=None,
    task_name='franka_custom',
    use_point_cloud=True,
    num_points=512,
    point_cloud_sampling_method='fps',
    camera_intrinsics={
        'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5,
        'width': 640, 'height': 480
    },
    workspace_bounds=[
        [0.3, 0.8],   # x bounds
        [-0.3, 0.3],  # y bounds
        [0.0, 0.5]    # z bounds
    ]
)

# Get validation dataset
val_dataset = dataset.get_validation_dataset()

# Get normalizer
normalizer = dataset.get_normalizer()

# Use with DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in dataloader:
    # batch contains:
    # batch['obs']['point_cloud']: Point cloud data (B, T, N, 3)
    # batch['obs']['agent_pos']: Robot state (B, T, D_state)
    # batch['action']: Actions (B, T, D_action)
    pass
```

### Configuration File

Create a YAML configuration file for your task:

```yaml
name: franka_custom

task_name: franka_custom

shape_meta: &shape_meta
  obs:
    point_cloud:
      shape: [512, 3]
      type: point_cloud
    agent_pos:
      shape: [7]  # Adjust based on your robot state dimension
      type: low_dim
  action:
    shape: [7]  # Adjust based on your action dimension

dataset:
  _target_: diffusion_policy_3d.dataset.franka_dataset.FrankaDataset
  dataset_path: /path/to/your/franka/dataset
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.02
  max_train_episodes: null
  task_name: franka_custom
  use_point_cloud: true
  num_points: 512
  point_cloud_sampling_method: 'fps'
  camera_intrinsics:
    fx: 525.0
    fy: 525.0
    cx: 319.5
    cy: 239.5
    width: 640
    height: 480
  workspace_bounds:
    - [0.3, 0.8]   # x bounds
    - [-0.3, 0.3]  # y bounds
    - [0.0, 0.5]   # z bounds
```

## Parameters

### FrankaDataset Parameters

- **dataset_path**: Path to your custom dataset directory
- **horizon**: Sequence length for training
- **pad_before/pad_after**: Padding for sequences
- **seed**: Random seed for reproducibility
- **val_ratio**: Fraction of data to use for validation
- **max_train_episodes**: Maximum number of training episodes (None for all)
- **task_name**: Name of the task
- **use_point_cloud**: Whether to generate point clouds from depth images
- **num_points**: Number of points to sample from each point cloud
- **point_cloud_sampling_method**: 'fps' (farthest point sampling) or 'uniform'
- **camera_intrinsics**: Camera intrinsic parameters for depth-to-point-cloud conversion
- **workspace_bounds**: Optional workspace bounds for filtering point clouds

### Camera Intrinsics

The camera intrinsics dictionary should contain:
- `fx`, `fy`: Focal lengths
- `cx`, `cy`: Principal point coordinates
- `width`, `height`: Image dimensions

### Workspace Bounds

Optional 3D workspace bounds to filter point clouds:
```python
workspace_bounds = [
    [x_min, x_max],
    [y_min, y_max],
    [z_min, z_max]
]
```

## Features

### RGBD to Point Cloud Conversion

- Automatically converts depth images to point clouds using Open3D
- Supports multiple camera views (left, right, wrist)
- Combines point clouds from all cameras
- Applies workspace filtering if specified

### Point Cloud Sampling

- **Farthest Point Sampling (FPS)**: More uniform distribution
- **Uniform Random Sampling**: Faster but less uniform

### Data Augmentation

- Point cloud sampling provides natural data augmentation
- Workspace filtering removes outliers

### Compatibility

- Same interface as `AdroitDataset`
- Works with existing training scripts
- Compatible with the 3D Diffusion Policy framework

## Tips

1. **Camera Calibration**: Ensure your camera intrinsics are accurate for good point cloud quality
2. **Workspace Bounds**: Use tight workspace bounds to remove background noise
3. **Sampling Method**: Use FPS for better point cloud quality, uniform for faster loading
4. **Number of Points**: Balance between detail (more points) and computational efficiency
5. **Memory Usage**: Consider your RAM when loading large datasets

## Troubleshooting

1. **Missing depth images**: The dataset will create dummy point clouds if depth images are missing
2. **Camera intrinsics**: Make sure the intrinsics match your depth image resolution
3. **Memory issues**: Reduce `num_points` or use fewer episodes
4. **Point cloud quality**: Check your depth images and camera calibration

## Example

See `example_franka_dataset.py` for a complete usage example.
