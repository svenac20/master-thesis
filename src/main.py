import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, BlendParams, RasterizationSettings, \
	PointLights, MeshRenderer, MeshRasterizer, SoftPhongShader

from PandaArm import PandaArm
from Utils import image_grid
from RobotMeshRenderer import RobotMeshRenderer

base_dir = os.path.abspath(".")
sys.path.append(base_dir)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	args = parser.parse_args("")
	args.base_dir = "/home/sscekic/master-thesis/src"
	args.use_gpu = False
	args.trained_on_multi_gpus = True
	args.keypoint_seg_model_path = os.path.join(args.base_dir, "weights/panda/panda-3cam_azure/net.pth")
	args.urdf_file = os.path.join(args.base_dir, "urdfs/Panda/panda.urdf")

	args.robot_name = 'Panda'  # "Panda" or "Baxter_left_arm"
	args.n_kp = 7
	args.height = 480
	args.width = 640
	args.fx, args.fy, args.px, args.py = 399.6578776041667, 399.4959309895833, 319.8955891927083, 244.0602823893229
	args.scale = 0.5  # scale the input image size to (320,240)

	# scale the camera parameters
	args.width = int(args.width * args.scale)
	args.height = int(args.height * args.scale)
	args.fx = args.fx * args.scale
	args.fy = args.fy * args.scale
	args.px = args.px * args.scale
	args.py = args.py * args.scale

	mesh_files = [base_dir + "/urdfs/Panda/meshes/visual/link0/link0.obj",
				  base_dir + "/urdfs/Panda/meshes/visual/link1/link1.obj",
				  base_dir + "/urdfs/Panda/meshes/visual/link2/link2.obj",
				  base_dir + "/urdfs/Panda/meshes/visual/link3/link3.obj",
				  base_dir + "/urdfs/Panda/meshes/visual/link4/link4.obj",
				  base_dir + "/urdfs/Panda/meshes/visual/link5/link5.obj",
				  base_dir + "/urdfs/Panda/meshes/visual/link6/link6.obj",
				  base_dir + "/urdfs/Panda/meshes/visual/link7/link7.obj",
				  base_dir + "/urdfs/Panda/meshes/visual/hand/hand.obj",
				  ]
	if torch.cuda.is_available():
		print("Using cuda")
		device = torch.device("cuda:0")
		torch.cuda.set_device(device)
	else:
			print(
					'Please note that NeRF is a resource-demanding method.'
					+ ' Running this notebook on CPU will be extremely slow.'
					+ ' We recommend running the example on a GPU'
					+ ' with at least 10 GB of memory.'
			)
			device = torch.device("cpu")

	R, T = look_at_view_transform(2.8, 0, 50)

	cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=30)
	blend_params = BlendParams(sigma=1e-8, gamma=1e-8)
	raster_settings = RasterizationSettings(
		image_size=(args.width, args.height),
		blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
		faces_per_pixel=100,
		max_faces_per_bin=100000,  # max_faces_per_bin=1000000,
	)
	lights = PointLights(device=device, location=((-2.0, -2.0, -2.0),))
	renderer = MeshRenderer(
		rasterizer=MeshRasterizer(
			cameras=cameras,
			raster_settings=raster_settings
		),
		shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
	)

 
	num_configs = 1000
	num_joints = 7
	angle_min = -2 * np.pi
	angle_max = 2 * np.pi
	# Generate the configurations
	configurations = np.random.uniform(low=angle_min, high=angle_max, size=(num_configs, num_joints))

	robot = PandaArm(args.urdf_file)
	robot_renderer = RobotMeshRenderer(robot, mesh_files, device)
	
	for configuration in configurations:
		robot_mesh = robot_renderer.get_robot_mesh(configuration)

		# Set batch size - this is the number of different viewpoints from which we want to render the mesh.
		batch_size = 4
		meshes = robot_mesh.extend(batch_size)

		# Get a batch of viewing angles.
		elev = torch.linspace(0, 180, batch_size)
		azim = torch.linspace(-180, 180, batch_size)

		R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
		cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=30)

		images = renderer(meshes, cameras=cameras, lights=lights)
		
		for index,image in enumerate(images, 1):
			print(image.shape)
			plt.figure(figsize=(10, 10))
			plt.imshow(image[..., :3].cpu().numpy())
			plt.axis("off")
			plt.savefig(f"generated-images/{index}/{configuration}.png")

		break
