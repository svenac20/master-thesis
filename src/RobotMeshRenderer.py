from os.path import exists
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

class RobotMeshRenderer():

	def __init__(self, robot, mesh_files, device):

		self.device = device
		self.robot = robot
		self.mesh_files = mesh_files
		self.preload_verts = []
		self.preload_faces = []

		for m_file in mesh_files:
			assert exists(m_file)
			preload_verts_i, preload_faces_idx_i, _ = load_obj(m_file)
			preload_faces_i = preload_faces_idx_i.verts_idx
			self.preload_verts.append(preload_verts_i)
			self.preload_faces.append(preload_faces_i)

	def get_robot_mesh(self, joint_angle):
		R_list, t_list = self.robot.get_joint_RT(joint_angle)
		assert len(self.mesh_files) == R_list.shape[0] and len(self.mesh_files) == t_list.shape[0]

		verts_list = []
		faces_list = []
		verts_rgb_list = []
		verts_count = 0
		for i in range(len(self.mesh_files)):
			verts_i = self.preload_verts[i]
			faces_i = self.preload_faces[i]

			R = torch.tensor(R_list[i], dtype=torch.float32)
			t = torch.tensor(t_list[i], dtype=torch.float32)
			verts_i = verts_i @ R.T + t
			faces_i = faces_i + verts_count

			verts_count += verts_i.shape[0]

			verts_list.append(verts_i.to(self.device))
			faces_list.append(faces_i.to(self.device))

			color = torch.rand(3)
			verts_rgb_i = torch.ones_like(verts_i) * color  # (V, 3)
			verts_rgb_list.append(verts_rgb_i.to(self.device))

		verts = torch.concat(verts_list, dim=0)
		faces = torch.concat(faces_list, dim=0)

		verts_rgb = torch.concat(verts_rgb_list, dim=0)[None]
		textures = Textures(verts_rgb=verts_rgb)

		# Create a Meshes object
		robot_mesh = Meshes(
			verts=[verts.to(self.device)],
			faces=[faces.to(self.device)],
			textures=textures
		)

		return robot_mesh
