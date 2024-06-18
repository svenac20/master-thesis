from torch.utils.data.dataset import Dataset

class EndEffectorDataset(Dataset):
    def __init__(self, end_effector_poses, solutions):
      self.end_effector_poses = end_effector_poses
      self.solutions = solutions
      # or use the RobertaTokenizer from `transformers` directly.

    def __len__(self):
      return len(self.end_effector_poses.shape[0])

    def __getitem__(self, i):
      return (self.end_effector_poses[i], self.solutions[i])
