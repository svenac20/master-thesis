import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, BlendParams, RasterizationSettings, \
	PointLights, MeshRenderer, MeshRasterizer, SoftPhongShader

from PandaArm import PandaArm
from Utils import image_grid
from src.RobotMeshRenderer import RobotMeshRenderer

base_dir = "/home/sscekic/master-thesis/src"

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	args = parser.parse_args("")
	args.base_dir = "/home/sscekic/CtRNet-robot-pose-estimation"
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

	device = 'cpu'
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

	robot = PandaArm(args.urdf_file)
	robot_renderer = RobotMeshRenderer(robot, mesh_files, device)
	configuration = np.array([-5.32755313, -4.51632518, 1.02188406, 5.39148447, 1.54878585, 3.39261642, -6.11622803])
	print(f"Currently rendering for configuration: {configuration}")
	joint_angles = configuration
	robot_mesh = robot_renderer.get_robot_mesh(joint_angles)

	# Set batch size - this is the number of different viewpoints from which we want to render the mesh.
	batch_size = 2

	# Create a batch of meshes by repeating the cow mesh and associated textures.
	# Meshes has a useful `extend` method which allows us do this very easily.
	# This also extends the textures.
	meshes = robot_mesh.extend(batch_size)

	# Get a batch of viewing angles.
	elev = torch.linspace(0, 180, batch_size)
	azim = torch.linspace(-180, 180, batch_size)

	# All the cameras helper methods support mixed type inputs and broadcasting. So we can
	# view the camera from the same distance and specify dist=2.7 as a float,
	# and then specify elevation and azimuth angles for each viewpoint as tensors.
	R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
	cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=30)

	# Move the light back in front of the cow which is facing the -z direction.
	lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)

	images = renderer(meshes, cameras=cameras, lights=lights)
	print(images.shape)
	image_grid(images.cpu().numpy(), rows=1, cols=2, rgb=True)
	plt.show()

