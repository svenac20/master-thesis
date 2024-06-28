import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, BlendParams, RasterizationSettings, \
	PointLights, MeshRenderer, MeshRasterizer, SoftPhongShader, SoftSilhouetteShader

from ImageCameraDataset import RobotImageDataset
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

	
	camera_angles = generate_camera_angles()
	images = []
	cameras = []
	silouhettes = []
	robot = PandaArm(args.urdf_file)
	robot_renderer = RobotMeshRenderer(robot, mesh_files, device)
	
	robot_mesh = robot_renderer.get_robot_mesh(configuration)
	verts = robot_mesh.verts_packed()
	N = verts.shape[0]
	center = verts.mean(0)
	scale = max((verts - center).abs().max(0)[0])
	robot_mesh.offset_verts_(-(center.expand(N, 3)))
	robot_mesh.scale_verts_((1.0 / float(scale)))

	for i, azim in enumerate(camera_angles):
		renderer_rgb, renderer_silhouette ,camera = create_renderer(args, device, 90, azim)

		image = renderer_rgb(robot_mesh)
		img = image[0, ..., :3].cpu().numpy()

		silhouette_images = renderer_silhouette(robot_mesh)
		silhouette_binary = (silhouette_images[..., 3] > 1e-4).float().squeeze(0)

		images.append(img)
		cameras.append(camera)
		silouhettes.append(silhouette_binary)

		plt.figure(figsize=(10, 10))
		print(f"Generating picture for configuration: {configuration}, camera angle: {azim}")
		plt.axis("off")
		plt.imshow(img)
		img_name = f"{args.images_folder}/{setImageName(configuration)}_angle_{azim}.png"
		plt.savefig(img_name, transparent=True, bbox_inches='tight', pad_inches=0)
		plt.close()

	dataset = RobotImageDataset(images=images, silhouette=silouhettes, cameras=cameras)
	torch.save(dataset, "images-camera-pairs.pth")

def generate_camera_angles(n=40):
	return np.random.uniform(-180, 180, size=n)


def create_renderer(args, device, elev, azim):
	R, T = look_at_view_transform(dist=2, elev=30, azim=azim, up=((0,0,1),))
	cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
	blend_params = BlendParams(sigma=1e-8, gamma=1e-8, background_color=(0.0,0.0,0.0))
	raster_settings = RasterizationSettings(
		image_size=(args.width, args.height),
		blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
		faces_per_pixel=100,
		max_faces_per_bin=100000,
	)
	lights = PointLights(device=device, location=((-2.0, -2.0, -2.0),))
	renderer_rgb = MeshRenderer(
		rasterizer=MeshRasterizer(
			cameras=cameras,
			raster_settings=raster_settings
		),
		shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params)
	)

	 # Rasterization settings for silhouette rendering
	sigma = 1e-4
	raster_settings_silhouette = RasterizationSettings(
			image_size=(args.width, args.height), 
			blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
			faces_per_pixel=100,
			max_faces_per_bin=100000,
	)
	# Silhouette renderer
	renderer_silhouette = MeshRenderer(
			rasterizer=MeshRasterizer(
					cameras=cameras, 
					raster_settings=raster_settings_silhouette
			),
			shader=SoftSilhouetteShader(),
	)

	return renderer_rgb, renderer_silhouette, cameras


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