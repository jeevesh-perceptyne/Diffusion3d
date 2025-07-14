import numpy as np
import time
from action_lipo import ActionLiPo
from franky import Robot, JointMotion , Gripper

# Parameters for LiPo smoothing
chunk = 50
blend = 10
time_delay = 3
lipo = ActionLiPo(chunk_size=chunk, blending_horizon=blend, len_time_delay=time_delay)

actions_path = "/home/prnuc/Documents/Jeevesh/Diffusion3d/3D-Diffusion-Policy/inferred_actions_infer.npy"
# Load actions and smooth all actions (not just a few)
actions = np.load(actions_path)
print("Actions loaded from path:", actions_path)
print("Shape of actions:", actions.shape)
actions = actions.astype(np.float64)
#print min,max ,mean of last column (gripper)
print("Gripper action - min:", np.min(actions[:, -1]), "max:", np.max(actions[:, -1]), "mean:", np.mean(actions[:, -1]))


# LiPo smoothing for all actions (overlapping chunks)
smoothed_actions = np.zeros_like(actions[:, :7])
count = np.zeros(actions.shape[0])
prev_chunk = None
for start in range(0, actions.shape[0] - chunk + 1, 1):  # stride 1 for full smoothing
    action_chunk = actions[start:start+chunk, :7]
    solved, _ = lipo.solve(action_chunk, prev_chunk, len_past_actions=blend if prev_chunk is not None else 0)
    if solved is not None:
        for i in range(chunk-blend):
            idx = start + i
            if idx < smoothed_actions.shape[0]:
                smoothed_actions[idx] += solved[i]
                count[idx] += 1
        prev_chunk = solved.copy()
    else:
        print(f"LiPo failed to solve chunk starting at {start}, using raw actions.")
        for i in range(chunk-blend):
            idx = start + i
            if idx < smoothed_actions.shape[0]:
                smoothed_actions[idx] += action_chunk[i]
                count[idx] += 1
        prev_chunk = action_chunk.copy()
# Average overlapping results
for i in range(smoothed_actions.shape[0]):
    if count[i] > 0:
        smoothed_actions[i] /= count[i]
    else:
        smoothed_actions[i] = actions[i, :7]
print("Smoothed actions shape:", smoothed_actions.shape)

# Franky robot execution
ip = "172.16.0.2"
robot = Robot(ip)
gripper = Gripper(ip)
joint_waypoints = []
for i in range(smoothed_actions.shape[0]):
    if(i% 10 == 0):
        print(f"Executing action {i+1}/{smoothed_actions.shape[0]}")
        action = smoothed_actions[i][:7]
        robot.move(JointMotion(action.tolist(),relative_dynamics_factor=0.01))
        # print("Gripper action:", actions[i][-1])
        try:
            gripper.move(0.08,1) if actions[i][-1] > 0.5 else gripper.grasp(0.0, 1, 1, epsilon_outer=1.0)
        except Exception as e:
            print(f"Error moving gripper: {e}")
        # time.sleep(0.05)


