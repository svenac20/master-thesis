import torch.nn as nn
import torch.nn.functional as F
import torch

device="cuda:1"
class TinyNeRF(nn.Module):
    def __init__(self, num_dof):
        super(TinyNeRF, self).__init__()
        self.spatial_fc = nn.Linear(3, 256, device=device)  # Spatial encoder
        self.dof_encoders = nn.ModuleList([nn.Linear(1, 256, device=device) for _ in range(num_dof)])  # DOF encoders
        self.fc1 = nn.Linear(256 + 256 * num_dof, 256, device=device)
        self.fc4 = nn.Linear(256, 256, device=device)
        self.fc5 = nn.Linear(256, 256, device=device)
        self.fc6 = nn.Linear(256, 256, device=device)
        self.fc7 = nn.Linear(256, 4, device=device)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, dof_values):
        batch_size = x.shape[0]
        spatial_out = F.relu(self.spatial_fc(x[...,:3]))
        dof_outs = [F.relu(encoder(dof_values[:, i:i+1]).expand(batch_size, -1)) for i, encoder in enumerate(self.dof_encoders)]
        dof_out = torch.cat(dof_outs, dim=-1)
        x = torch.cat([spatial_out, dof_out], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        x = self.sigmoid(x)
        return x