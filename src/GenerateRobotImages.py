import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, BlendParams, RasterizationSettings, \
	PointLights, MeshRenderer, MeshRasterizer, SoftPhongShader

from PandaArm import PandaArm
from Utils import setImageName
from RobotMeshRenderer import RobotMeshRenderer

def generate_3d_images(args, configuration):
	mesh_files = get_mesh_files(args)

	if torch.cuda.is_available():
		device = torch.device("cuda:1")
		torch.cuda.set_device(device)
	else:
			device = torch.device("cpu")

	renderer, cameras = create_renderer(args, device, 90, 0)
	robot = PandaArm(args.urdf_file)
	robot_renderer = RobotMeshRenderer(robot, mesh_files, device)

	robot_mesh = robot_renderer.get_robot_mesh(configuration)
	images = renderer(robot_mesh)

	plt.figure(figsize=(10, 10))
	plt.axis("off")
	#return images[0,..., :3], cameras
	print(f"Generating picture for configuration: {configuration}")
	plt.savefig(f"{args.images_folder}/{setImageName(configuration)}.png", transparent=True, bbox_inches='tight', pad_inches=0)
	plt.close()

def generate_camera_angles():
	angles = []
	for i in range(20):
		elev = np.random.uniform(0, 360)
		azim = np.random.uniform(0, 360)
		angles.append((elev, azim))
	return angles



def create_renderer(args, device, elev, azim):
	camera_distance = np.random.uniform(1.3, 2)
	R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim)
	cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
	blend_params = BlendParams(sigma=1e-8, gamma=1e-8)
	raster_settings = RasterizationSettings(
		image_size=(args.width, args.height),
		blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
		faces_per_pixel=100,
		max_faces_per_bin=100000,
	)
	lights = PointLights(device=device, location=((-2.0, -2.0, -2.0),))
	renderer = MeshRenderer(
		rasterizer=MeshRasterizer(
			cameras=cameras,
			raster_settings=raster_settings
		),
		shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
	)

	return renderer, cameras


def get_mesh_files(args):
	base_dir = args.base_dir
	return [base_dir + "/urdfs/Panda/meshes/visual/link0/link0.obj",
					base_dir + "/urdfs/Panda/meshes/visual/link1/link1.obj",
					base_dir + "/urdfs/Panda/meshes/visual/link2/link2.obj",
					base_dir + "/urdfs/Panda/meshes/visual/link3/link3.obj",
					base_dir + "/urdfs/Panda/meshes/visual/link4/link4.obj",
					base_dir + "/urdfs/Panda/meshes/visual/link5/link5.obj",
					base_dir + "/urdfs/Panda/meshes/visual/link6/link6.obj",
					base_dir + "/urdfs/Panda/meshes/visual/link7/link7.obj",
					base_dir + "/urdfs/Panda/meshes/visual/hand/hand.obj",
					]