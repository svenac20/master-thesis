import os
import matplotlib.pyplot as plt
from itertools import chain, combinations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)

import os

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    look_at_view_transform,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
)



def create_powerset(iterable):
	s = list(iterable)
	return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generate_configurations(number_per_powerset, number_base_rotation):
	powersets = create_powerset([1,2,3,4,5,6])
	joint_limits = [
			(-2.8973, 2.8973),    
			(-1.7628, 1.7628),   
			(-2.8973, 2.8973), 
			(-3.0718, -0.0698),
			(-2.8973, 2.8973),
			(-0.0175, 3.7525),
			(-2.8973, 2.8973)
		]
	all_configurations = np.empty((0,7))
	for powerset in powersets:
		if powerset == ():
				continue

		# set temporary limits for current subset
		temp_limits = np.array([(0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)])
		for index in powerset:
				temp_limits[index] = joint_limits[index]

		temp_configurations = np.array([
			np.random.uniform(low=min_angle, high=max_angle, size=number_per_powerset)
			for min_angle, max_angle in temp_limits
		]).T

		joint_0_configurations = np.random.uniform(low=joint_limits[0][0], high=joint_limits[0][1], size=number_base_rotation)

		configurations = np.empty((0,7))
		for configuration in temp_configurations:
			for joint_0_configuration in joint_0_configurations:
				configuration[0] = joint_0_configuration;
				configuration = np.where(np.abs(configuration) < 1e-6, 0, configuration)
				configurations = np.vstack([configurations, configuration])

		all_configurations = np.vstack([all_configurations, configurations])

	return all_configurations


def setImageName(configuration):
	array_as_strings = configuration.astype(str)
	return ";".join(array_as_strings)

def getConfigurationFromImageName(s, dtype=float):
	extension = s.rfind(".")
	s = s[:extension]
	str_list = s.split(';')
		
	array = np.array(str_list, dtype=dtype)
		
	return array


class TensorImageDataset(Dataset):
	def __init__(self, images_tensor, configurations):
			self.images = images_tensor
			self.configurations = configurations

	def __len__(self):
			return self.images.shape[0]

	def __getitem__(self, idx):
			return self.images[idx], self.configurations[idx]
		

def getImagesDataloader(path, batch_size=16):
	img_names = os.listdir(path)
	width = 577
	height = 770

	images = np.empty((len(img_names), height, width, 3))
	configurations = np.empty((len(img_names),1, 7))
	for idx, name in enumerate(img_names):
		configurations[idx] = [getConfigurationFromImageName(name)]
		img_name = path + name
		# Use you favourite library to load the image
		image = plt.imread(img_name)
		image = image[:, :, :3]
		images[idx] = image

	dataset = TensorImageDataset(images, configurations)
	return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def generate_rays_pytorch3d(cameras, image_height, image_width):
	"""
	Generate ray origins and directions for each pixel in the image using PyTorch3D camera.

	Args:
			cameras (PerspectiveCameras): PyTorch3D camera object.
			image_height (int): Height of the image.
			image_width (int): Width of the image.

	Returns:
			ray_origins (torch.Tensor): Ray origins in world space (image_height * image_width, 3).
			ray_directions (torch.Tensor): Ray directions in world space (image_height * image_width, 3).
	"""
	# Create a mesh grid of pixel coordinates
	i, j = torch.meshgrid(torch.linspace(0, image_width - 1, image_width),
												torch.linspace(0, image_height - 1, image_height))
	i = i.t()
	j = j.t()

	# Normalize pixel coordinates to range [-1, 1]
	pixel_coords = torch.stack([i, j], dim=-1).float()
	pixel_coords[..., 0] = (pixel_coords[..., 0] / (image_width - 1)) * 2 - 1
	pixel_coords[..., 1] = (pixel_coords[..., 1] / (image_height - 1)) * 2 - 1

	# Create homogeneous coordinates (H, W, 3)
	pixel_coords = torch.cat([pixel_coords, torch.ones_like(pixel_coords[..., :1])], dim=-1)
	pixel_coords = pixel_coords.view(-1, 3)  # Reshape to (H * W, 3)

	# Transform pixel coordinates to camera coordinates
	ray_directions = cameras.unproject_points(pixel_coords)  # (H * W, 3)

	# Get camera center in world coordinates
	camera_center = cameras.get_camera_center().squeeze(0)  # (3)

	# Compute ray directions
	ray_directions = ray_directions - camera_center  # (H * W, 3)
	ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)  # Normalize directions

	# The origin of all rays is the camera position in world coordinates
	ray_origins = camera_center.expand(ray_directions.shape)

	return ray_origins, ray_directions


def generate_random_pose():
	# Generate a random translation vector (x, y, z)
	translation = np.random.uniform(-1, 1, 3)
	
	# Generate a random rotation matrix
	rotation = R.random().as_matrix()
	
	# Create a 4x4 homogeneous transformation matrix
	pose_matrix = np.eye(4)
	pose_matrix[:3, :3] = rotation
	pose_matrix[:3, 3] = translation
	
	return pose_matrix

def generate_end_effector_poses(number_of_poses):
	poses = []
	for _ in range(number_of_poses):
			pose_matrix = generate_random_pose()
			poses.append(pose_matrix)
	
	poses = np.array(poses)
	return torch.as_tensor(poses, dtype=torch.float)

def generate_base_rotation_joint_angles(number_of_rotations):
	joints = torch.zeros((number_of_rotations, 7))
	base_rotations = np.random.uniform(low=-2.8973, high=2.8973, size=number_of_rotations)
	joints[:, 0] = torch.as_tensor(base_rotations)

	return joints

def getBatchCameraFromIndexes(target_cameras, batch_idx, device):
	R_list = []
	T_list = []
	znear_list = []
	zfar_list =[]
	aspect_ratio_list = []
	fov_list = []

	for index in batch_idx:
		camera = target_cameras[index]
		R_list.append(camera.R)
		T_list.append(camera.T)
		znear_list.append(camera.znear)
		zfar_list.append(camera.zfar)
		aspect_ratio_list.append(camera.aspect_ratio)
		fov_list.append(camera.fov)

	R_tensor = torch.cat(R_list, dim=0)
	T_tensor = torch.cat(T_list, dim=0)
	znear = torch.cat(znear_list, dim=0)
	zfar = torch.cat(zfar_list, dim=0)
	aspect_ratio = torch.cat(aspect_ratio_list, dim=0)
	fov = torch.cat(fov_list, dim=0)

	return FoVPerspectiveCameras(
				R = R_tensor,
				T = T_tensor,
				znear = znear,
        zfar = zfar,
        aspect_ratio = aspect_ratio,
        fov = fov,
        device = device
		)

def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()