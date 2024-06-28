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
from Utils import getBatchCameraFromIndexes, image_grid
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
target_silhouettes = []
a = 1
for image,silhouette,camera in dataset:
  target_images.append(image)
  target_cameras.append(camera)
  target_silhouettes.append(silhouette.cpu().numpy())

target_images = torch.as_tensor(target_images).to(device)
target_silhouettes = torch.as_tensor(target_silhouettes).to(device)
renderer_grid = renderer_grid.to(device)
renderer_mc = renderer_mc.to(device)


# Instantiate the radiance field model.
neural_radiance_field = NeuralRadianceField().to(device)

# Instantiate the Adam optimizer. We set its master learning rate to 1e-3.
lr = 1e-3
optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr)

# We sample 6 random cameras in a minibatch. Each camera
# emits raysampler_mc.n_pts_per_image rays.
batch_size = 10

# Init the loss history buffers.
n_iter = 20000
loss_history_color, loss_history_sil = [], []

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

  silhouettes_at_rays = sample_images_at_mc_locs(
        target_silhouettes[batch_idx, ..., None], 
        sampled_rays.xys
    )
  sil_err = huber(
        rendered_silhouettes, 
        silhouettes_at_rays,
    ).abs().mean()

  colors_at_rays = sample_images_at_mc_locs(
        target_images[batch_idx].to(device), 
        sampled_rays.xys
    )
  color_err = huber(
      rendered_images, 
      colors_at_rays,
  ).abs().mean()
  
  loss = color_err + sil_err
    
  loss_history_color.append(float(color_err))
  loss_history_sil.append(float(sil_err))

  if iteration % 10 == 0:
        print(
            f'Iteration {iteration:05d}:'
            + f' loss color = {float(color_err):1.2e}'
            + f' loss sill = {float(sil_err):1.2e}'
        )
  loss.backward()
  optimizer.step()


  if iteration % 500 == 0:
        show_idx = torch.randperm(len(target_cameras))[:1]
        show_cameras = getBatchCameraFromIndexes(target_cameras, show_idx, device)
        show_full_render(
            neural_radiance_field,
            renderer_grid=renderer_grid,
            camera=show_cameras,
            target_image=target_images[show_idx][0].to(device),
            target_silhouette=target_silhouettes[show_idx][0].to(device),
            loss_history_color=loss_history_color,
            loss_history_sil=loss_history_sil
        )

torch.save(neural_radiance_field.state_dict(), "nerf-model-final.pth")

def generate_rotating_nerf(neural_radiance_field, n_frames = 50):
    logRs = torch.zeros(n_frames, 3, device=device)
    logRs[:, 1] = torch.linspace(-3.14, 3.14, n_frames, device=device)
    Rs = so3_exp_map(logRs)
    Ts = torch.zeros(n_frames, 3, device=device)
    Ts[:, 2] = 2.7
    frames = []
    print('Rendering rotating NeRF ...')
    for R, T in zip(tqdm(Rs), Ts):
        camera = FoVPerspectiveCameras(
            R=R[None], 
            T=T[None], 
            znear=target_cameras[0].znear,
            zfar=target_cameras[0].zfar,
            aspect_ratio=target_cameras[0].aspect_ratio,
            fov=target_cameras[0].fov,
            device=device,
        )
        # Note that we again render with `NDCMultinomialRaysampler`
        # and the batched_forward function of neural_radiance_field.
        frames.append(
            renderer_grid(
                cameras=camera, 
                volumetric_function=neural_radiance_field.batched_forward,
            )[0][..., :3]
        )
    return torch.cat(frames)
    
with torch.no_grad():
    rotating_nerf_frames = generate_rotating_nerf(neural_radiance_field, n_frames=3*5)
    
image_grid(rotating_nerf_frames.clamp(0., 1.).cpu().numpy(), rows=3, cols=5, rgb=True, fill=True)
plt.show()
