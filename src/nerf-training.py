import os
import sys
import time
import json
import glob
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython import display
from tqdm.notebook import tqdm

# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from Utils import getBatchCameraFromIndexes
from generate_cow_renders import generate_cow_renders
from nerf import NeuralRadianceField, huber, sample_images_at_mc_locs, show_full_render

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
else:
    print(
        'Please note that NeRF is a resource-demanding method.'
        + ' Running this notebook on CPU will be extremely slow.'
        + ' We recommend running the example on a GPU'
        + ' with at least 10 GB of memory.'
    )
    device = torch.device("cpu")
render_size = 200*2

# Our rendered scene is centered around (0,0,0) 
# and is enclosed inside a bounding box
# whose side is roughly equal to 3.0 (world units).
volume_extent_world = 3.0

# 1) Instantiate the raysamplers.

# Here, NDCMultinomialRaysampler generates a rectangular image
# grid of rays whose coordinates follow the PyTorch3D
# coordinate conventions.
raysampler_grid = NDCMultinomialRaysampler(
    image_height=render_size,
    image_width=render_size,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

# MonteCarloRaysampler generates a random subset 
# of `n_rays_per_image` rays emitted from the image plane.
raysampler_mc = MonteCarloRaysampler(
    min_x = -1.0,
    max_x = 1.0,
    min_y = -1.0,
    max_y = 1.0,
    n_rays_per_image=750,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

# 2) Instantiate the raymarcher.
# Here, we use the standard EmissionAbsorptionRaymarcher 
# which marches along each ray in order to render
# the ray into a single 3D color vector 
# and an opacity scalar.
raymarcher = EmissionAbsorptionRaymarcher()

# Finally, instantiate the implicit renders
# for both raysamplers.
renderer_grid = ImplicitRenderer(
    raysampler=raysampler_grid, raymarcher=raymarcher,
)
renderer_mc = ImplicitRenderer(
    raysampler=raysampler_mc, raymarcher=raymarcher,
)
dataset = torch.load("images-camera-pairs.pth")

target_images = []
target_cameras = []
for image,camera in dataset:
  target_images.append(image)
  target_cameras.append(camera)

target_images = torch.as_tensor(target_images).to(device)
renderer_grid = renderer_grid.to(device)
renderer_mc = renderer_mc.to(device)


# Set the seed for reproducibility
torch.manual_seed(1)

# Instantiate the radiance field model.
neural_radiance_field = NeuralRadianceField().to(device)

# Instantiate the Adam optimizer. We set its master learning rate to 1e-3.
lr = 1e-3
optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr)

# We sample 6 random cameras in a minibatch. Each camera
# emits raysampler_mc.n_pts_per_image rays.
batch_size = 6

# Init the loss history buffers.
loss_arr = []
n_iter = 10000
criterion = nn.MSELoss()

for iteration in range(n_iter):
  if iteration == round(n_iter * 0.75):
        print('Decreasing LR 10-fold ...')
        optimizer = torch.optim.Adam(
            neural_radiance_field.parameters(), lr=lr * 0.1
        )
  optimizer.zero_grad()
    
  # Sample random batch indices.
  batch_idx = torch.randperm(len(target_cameras))[:batch_size]
  batch_cameras = getBatchCameraFromIndexes(target_cameras, batch_idx, device)

  rendered_images_silhouettes, sampled_rays = renderer_mc(
        cameras=batch_cameras, 
        volumetric_function=neural_radiance_field
    )
  rendered_images, rendered_silhouettes = (
      rendered_images_silhouettes.split([3, 1], dim=-1)
  )

  colors_at_rays = sample_images_at_mc_locs(
        target_images[batch_idx].to(device), 
        sampled_rays.xys
    )
  color_err = huber(
      rendered_images, 
      colors_at_rays,
  ).abs().mean()
  
  loss = color_err
    
  loss_arr.append(loss.item())
  if iteration % 10 == 0:
        print(
            f'Iteration {iteration:05d}:'
            + f' loss color = {float(color_err):1.2e}'
        )
  loss.backward()
  optimizer.step()


  torch.cuda.empty_cache()
  if iteration % 100 == 0:
        show_idx = torch.randperm(len(target_cameras))[:1]
        show_cameras = getBatchCameraFromIndexes(target_cameras, show_idx, device)
        show_full_render(
            neural_radiance_field,
            renderer_grid=renderer_grid,
            camera=show_cameras,
            target_image=target_images[show_idx][0].to(device),
            loss_history_color=loss_arr
        )