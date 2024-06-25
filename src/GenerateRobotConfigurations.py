import numpy as np
from Utils import generate_end_effector_poses, generate_base_rotation_joint_angles
from EndEffectorDataset import EndEffectorDataset
import inverse_kinematics_solver.franka_ik as ik
import torch

def generate_robot_configurations(N, datasetPath, seed=1234, num_base_rotations=4):
  np.random.seed(seed)

  # [N, 4, 4]
  end_effector_poses = generate_end_effector_poses(N)
  # we generate 4 different base joint rotations. Other joint angles are not used
  # [4, 7]
  base_rotations = generate_base_rotation_joint_angles(num_base_rotations)

  valid_poses = []
  result_configurations = []

  for end_effector_pose in end_effector_poses:
    # random joint angle for last joint
    last_joint_angle = np.random.uniform(low=-2.8973, high=2.8973, size=1)
    end_effector_reshaped = end_effector_pose.reshape(16)

    configuration = ik.ik_solver(end_effector_reshaped.tolist(), last_joint_angle[0], base_rotations[0].tolist())
    configuration = torch.as_tensor(configuration)
    filtered_configurations = configuration[~torch.any(configuration.isnan(),dim=1)]

    for filtered_configuration in filtered_configurations:
      valid_poses.append(end_effector_pose)
      result_configurations.append(filtered_configuration)

  valid_poses_tensor = torch.stack(valid_poses)
  result_configurations_tensor = torch.stack(result_configurations)

  dataset = EndEffectorDataset(valid_poses_tensor, result_configurations_tensor)
  torch.save(dataset, datasetPath)