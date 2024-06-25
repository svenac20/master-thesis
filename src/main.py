import argparse
import os
import sys
import torch
import math
import torch.nn as nn
from pytorch3d.transforms import euler_angles_to_matrix
from GenerateRobotConfigurations import generate_robot_configurations
from GenerateRobotImages import generate_3d_images
from nerf import TinyNeRF
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.abspath(".") + "/src"
sys.path.append(base_dir)
device="cuda:1"

def get_rays(H, W, K, extrinsics):
		i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=device), 
													torch.arange(H, dtype=torch.float32, device=device), indexing='ij')
		i = i.t()
		j = j.t()
		dirs = torch.stack([(i - K[0, 2]) / K[0, 0], (j - K[1, 2]) / K[1, 1], torch.ones_like(i)], -1)
		dirs = dirs.reshape(-1, 3).cuda(device)
		rays_d = torch.sum(dirs[..., None, :] * extrinsics[:3, :3], -1)
		rays_o = extrinsics[:3, 3].expand(rays_d.shape)
		
		return rays_o, rays_d

def get_camera_extrinsics(camera):
		R, T = camera.R, camera.T
		extrinsics = torch.eye(4)
		extrinsics[:3, :3] = R
		extrinsics[:3, 3] = T
		return extrinsics

def testModel(args):
	joint_angles = torch.as_tensor(np.array([ 0.0200, -0.9641, -0.0662, -2.7979, -0.0469,  1.9289,  0.9137]), dtype=torch.float)
	image, camera = generate_3d_images(args,joint_angles.numpy())
	H, W = image.shape[0:2]
	extrinsics = get_camera_extrinsics(camera).cuda(device=device)
	K = camera.get_full_projection_transform().get_matrix()[0]
	dof_values = joint_angles.expand([H*W, 7])
	
	model = TinyNeRF(num_dof=7).cuda()
	model.load_state_dict(torch.load('tinynerf_model.pth'))
	model.eval()
	rays_o, rays_d = get_rays(H, W, K, extrinsics)
	rays = torch.cat([rays_o, rays_d], dim=-1).cuda(device)
	with torch.no_grad():
		rendered_image = model(rays, dof_values.cuda(device))
	# Reshape and save the rendered image
	rendered_image = rendered_image[..., :3].reshape(H, W, 3).cpu().numpy()
	f, axarr = plt.subplots(2,1) 

	axarr[0].imshow(rendered_image)
	axarr[1].imshow(image[..., :3].cpu().numpy())

	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', nargs='?', const=base_dir, type=str, default=base_dir)
	parser.add_argument('--width', nargs='?', const=400, type=int, default=400)
	parser.add_argument('--height', nargs='?', const=400, type=int, default=400)
	parser.add_argument('--urdf_file', nargs='?', const=os.path.join(base_dir, "urdfs/Panda/panda.urdf"), type=str, default=os.path.join(base_dir, "urdfs/Panda/panda.urdf"))
	parser.add_argument('--dataset', nargs='?', const=os.path.join(base_dir, "generated-dataset/ee-configurations.pt"), type=str, default=os.path.join(base_dir, "generated-dataset/ee-configurations.pt"))
	parser.add_argument('--images_folder', nargs='?', const=os.path.join(base_dir, "generated-images"), type=str, default=os.path.join(base_dir, "generated-images"))
	args = parser.parse_args()

	if (not os.path.exists(os.path.join(base_dir, args.dataset))):
		generate_robot_configurations(5000, args.dataset)

	# if (os.path.exists(os.path.join("tinynerf_model.pth"))):
	# 	testModel(args)

	dataset = torch.load(args.dataset)
	model = TinyNeRF(num_dof=7)
	optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
	criterion = nn.MSELoss()
	for (ee_pose, configuration) in dataset:
		image, camera = generate_3d_images(args,configuration.numpy())
		H, W = image.shape[0:2]
		extrinsics = get_camera_extrinsics(camera).cuda(device=device)
		K = camera.get_full_projection_transform().get_matrix()[0]

		rays_o, rays_d = get_rays(H, W, K, extrinsics)
		rays = torch.cat([rays_o, rays_d], dim=-1).cuda(device)
		
		# Forward pass
		configuration = configuration.reshape(1,7).cuda(device)
		raw = model(rays, configuration)
		rgb = raw[..., :3].reshape(H, W, 3).cuda(device)
		
		# f, axarr = plt.subplots(2,1) 

		# # use the created array to output your multiple images. In this case I have stacked 2 images vertically
		# axarr[0].imshow(rgb.cpu().detach().numpy())
		# axarr[1].imshow(image.cpu().numpy())

		# plt.show()
		# plt.close()
		# Compute loss
		loss = criterion(rgb, image)
		
		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		print(f"Loss:{loss.item()}")

	torch.save(model.state_dict(), 'tinynerf_model.pth')



