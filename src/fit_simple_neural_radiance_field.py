
# coding: utf-8

# In[ ]:


# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.


# # Fit a simple Neural Radiance Field via raymarching
# 
# This tutorial shows how to fit Neural Radiance Field given a set of views of a scene using differentiable implicit function rendering.
# 
# More specifically, this tutorial will explain how to:
# 1. Create a differentiable implicit function renderer with either image-grid or Monte Carlo ray sampling.
# 2. Create an Implicit model of a scene.
# 3. Fit the implicit function (Neural Radiance Field) based on input images using the differentiable implicit renderer. 
# 4. Visualize the learnt implicit function.
# 
# Note that the presented implicit model is a simplified version of NeRF:<br>
# _Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, ECCV 2020._
# 
# The simplifications include:
# * *Ray sampling*: This notebook does not perform stratified ray sampling but rather ray sampling at equidistant depths.
# * *Rendering*: We do a single rendering pass, as opposed to the original implementation that does a coarse and fine rendering pass.
# * *Architecture*: Our network is shallower which allows for faster optimization possibly at the cost of surface details.
# * *Mask loss*: Since our observations include segmentation masks, we also optimize a silhouette loss that forces rays to either get fully absorbed inside the volume, or to completely pass through it.
# 

# ## 0. Install and Import modules
# Ensure `torch` and `torchvision` are installed. If `pytorch3d` is not installed, install it using the following cell:

# In[ ]:


import os
import sys
import torch
need_pytorch3d=False
import pytorch3d


# In[ ]:


# %matplotlib inline
# %matplotlib notebook
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
    look_at_view_transform
)

# obtain the utilized device
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


# In[ ]:


# get_ipython().system('wget https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/docs/tutorials/utils/plot_image_grid.py')
# get_ipython().system('wget https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/docs/tutorials/utils/generate_cow_renders.py')
# from plot_image_grid import image_grid
# from generate_cow_renders import generate_cow_renders


# OR if running locally uncomment and run the following cell:

# In[ ]:


from generate_cow_renders import generate_cow_renders
from Utils import generate_rays_pytorch3d, getImagesDataloader, image_grid


# ## 1. Generate images of the scene and masks
# 
# The following cell generates our training data.
# It renders the cow mesh from the `fit_textured_mesh.ipynb` tutorial from several viewpoints and returns:
# 1. A batch of image and silhouette tensors that are produced by the cow mesh renderer.
# 2. A set of cameras corresponding to each render.
# 
# Note: For the purpose of this tutorial, which aims at explaining the details of implicit rendering, we do not explain how the mesh rendering, implemented in the `generate_cow_renders` function, works. Please refer to `fit_textured_mesh.ipynb` for a detailed explanation of mesh rendering.

# In[ ]:

images_data_loader = getImagesDataloader("src/generated-images-powerset-2/")

# ## 2. Initialize the implicit renderer
# 
# The following initializes an implicit renderer that emits a ray from each pixel of a target image and samples a set of uniformly-spaced points along the ray. At each ray-point, the corresponding density and color value is obtained by querying the corresponding location in the neural model of the scene (the model is described & instantiated in a later cell).
# 
# The renderer is composed of a *raymarcher* and a *raysampler*.
# - The *raysampler* is responsible for emitting rays from image pixels and sampling the points along them. Here, we use two different raysamplers:
#     - `MonteCarloRaysampler` is used to generate rays from a random subset of pixels of the image plane. The random subsampling of pixels is carried out during **training** to decrease the memory consumption of the implicit model.
#     - `NDCMultinomialRaysampler` which follows the standard PyTorch3D coordinate grid convention (+X from right to left; +Y from bottom to top; +Z away from the user). In combination with the implicit model of the scene, `NDCMultinomialRaysampler` consumes a large amount of memory and, hence, is only used for visualizing the results of the training at **test** time.
# - The *raymarcher* takes the densities and colors sampled along each ray and renders each ray into a color and an opacity value of the ray's source pixel. Here we use the `EmissionAbsorptionRaymarcher` which implements the standard Emission-Absorption raymarching algorithm.

# In[ ]:




# ## 3. Define the neural radiance field model
# 
# In this cell we define the `NeuralRadianceField` module, which specifies a continuous field of colors and opacities over the 3D domain of the scene.
# 
# The `forward` function of `NeuralRadianceField` (NeRF) receives as input a set of tensors that parametrize a bundle of rendering rays. The ray bundle is later converted to 3D ray points in the world coordinates of the scene. Each 3D point is then mapped to a harmonic representation using the `HarmonicEmbedding` layer (defined in the next cell). The harmonic embeddings then enter the _color_ and _opacity_ branches of the NeRF model in order to label each ray point with a 3D vector and a 1D scalar ranging in [0-1] which define the point's RGB color and opacity respectively.
# 
# Since NeRF has a large memory footprint, we also implement the `NeuralRadianceField.forward_batched` method. The method splits the input rays into batches and executes the `forward` function for each batch separately in a for loop. This lets us render a large set of rays without running out of GPU memory. Standardly, `forward_batched` would be used to render rays emitted from all pixels of an image in order to produce a full-sized render of a scene.
# 

# In[ ]:
class CoordinatesEncoder(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_size),
            torch.nn.Linear(input_size, output_size),
            torch.nn.ReLU()
        ).to("cuda:1")
    
    def forward(self, x):
        return self.network(x)

class ConfigurationEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_dim_size, output_size):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_size),
            torch.nn.Linear(input_size, output_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_size, hidden_dim_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_size, output_size),
            torch.nn.ReLU(),
        ).to("cuda:1")
    
    def forward(self, x):
        # x is [16,750,128]
        return self.network(x)

class NeuralRadianceField(torch.nn.Module):
    def __init__(self, n_hidden_neurons=256, encoders_output=128):
        super().__init__()
        """
        Args:
            n_hidden_neurons: The number of hidden units in the
                fully connected layers of the MLPs of the model.
        """
        # self.mlp is a simple 2-layer multi-layer perceptron
        # which converts the input per-point harmonic embeddings
        # to a latent representation.
        self.coordinate_encoder = torch.nn.ModuleList([CoordinatesEncoder(input_size=1, output_size=encoders_output) for _ in range(3)])
        
        self.configuration_encoder = torch.nn.ModuleList([ConfigurationEncoder(input_size=1, hidden_dim_size=n_hidden_neurons, output_size=encoders_output) for _ in range(7)])        

        self.group_coordinate_encoder = torch.nn.Sequential(
            torch.nn.Linear(3 * encoders_output, n_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_neurons, encoders_output),
            torch.nn.ReLU(),
        ).to("cuda:1")

        self.group_configuration_encoder = torch.nn.Sequential(
            torch.nn.Linear(7 * encoders_output, n_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_neurons, encoders_output),
            torch.nn.ReLU(),
        ).to("cuda:1")
        
        # The density layer converts the features of self.mlp
        # to a 1D density value representing the raw opacity
        # of each point.
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(2 * encoders_output, n_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.ReLU(),
        ).to("cuda:1")

        # Given features predicted by self.mlp, self.color_layer
        # is responsible for predicting a 3-D per-point vector
        # that represents the RGB color of the point.
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(2 * encoders_output, n_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
            # To ensure that the colors correctly range between [0-1],
            # the layer is terminated with a sigmoid layer.
        ).to("cuda:1")  
                
    def forward(
        self, 
        ray_bundle,
        configuration,
        **kwargs,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's 
        RGB color and opacity respectively.
        
        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.

        # [batch_size x num_points_per_ray x width x color]
        # [1, 750, 128, 3]
        rays_points_world = ray_bundle_to_ray_points(ray_bundle).to("cuda:1")
        
        #extend to batch size for easier calculations
        # [16, 750, 128, 3]
        rays_points_world = rays_points_world.repeat(16,1,1,1)

        # encode coordinates
        encoded_coordinates = []
        for i in range(3):
            encoded_coordinates.append(self.coordinate_encoder[i](rays_points_world[..., i:i+1].permute(0,3,1,2)).to("cuda:1"))
        encoded_coordinates = torch.stack(encoded_coordinates, dim=-1)

        encoded_configurations = []
        for i in range(7):
            encoded_configurations.append(self.configuration_encoder[i](configuration[i]))
        encoded_configurations = torch.stack(encoded_configurations, dim=-1)
        
        group_encoded_coordinates = self.group_coordinate_encoder(encoded_coordinates)
        group_encoded_configurations = self.group_configuration_encoder(encoded_configurations)

        group_encoded = torch.cat((group_encoded_coordinates, group_encoded_configurations), -1)
        
        density = self.density_layer(group_encoded)
        color = self.color_layer(group_encoded)
    
        return density, color
    
# ## 4. Helper functions
# 
# In this function we define functions that help with the Neural Radiance Field optimization.

# In[ ]:


def sample_images_at_mc_locs(target_images, sampled_rays_xy):
    """
    Given a set of Monte Carlo pixel locations `sampled_rays_xy`,
    this method samples the tensor `target_images` at the
    respective 2D locations.
    
    This function is used in order to extract the colors from
    ground truth images that correspond to the colors
    rendered using `MonteCarloRaysampler`.
    """
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]
    # In order to sample target_images, we utilize
    # the grid_sample function which implements a
    # bilinear image sampler.
    # Note that we have to invert the sign of the 
    # sampled ray positions to convert the NDC xy locations
    # of the MonteCarloRaysampler to the coordinate
    # convention of grid_sample.
    images_sampled = torch.nn.functional.grid_sample(
        target_images.permute(0, 3, 1, 2), 
        -sampled_rays_xy.view(ba, -1, 1, 2),  # note the sign inversion
        align_corners=True
    )
    return images_sampled.permute(0, 2, 3, 1).view(
        ba, *spatial_size, dim
    )

def show_full_render(
    neural_radiance_field, camera,
    target_image, target_silhouette,
    loss_history_color, loss_history_sil,
):
    """
    This is a helper function for visualizing the
    intermediate results of the learning. 
    
    Since the `NeuralRadianceField` suffers from
    a large memory footprint, which does not let us
    render the full image grid in a single forward pass,
    we utilize the `NeuralRadianceField.batched_forward`
    function in combination with disabling the gradient caching.
    This chunks the set of emitted rays to batches and 
    evaluates the implicit function on one batch at a time
    to prevent GPU memory overflow.
    """
    
    # Prevent gradient caching.
    with torch.no_grad():
        # Render using the grid renderer and the
        # batched_forward function of neural_radiance_field.
        rendered_image_silhouette, _ = renderer_grid(
            cameras=camera, 
            volumetric_function=neural_radiance_field.batched_forward
        )
        # Split the rendering result to a silhouette render
        # and the image render.
        rendered_image, rendered_silhouette = (
            rendered_image_silhouette[0].split([3, 1], dim=-1)
        )
        
    # Generate plots.
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax = ax.ravel()
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    ax[0].plot(list(range(len(loss_history_color))), loss_history_color, linewidth=1)
    ax[1].imshow(clamp_and_detach(rendered_image))
    ax[2].imshow(clamp_and_detach(rendered_silhouette[..., 0]))
    ax[3].plot(list(range(len(loss_history_sil))), loss_history_sil, linewidth=1)
    ax[4].imshow(clamp_and_detach(target_image))
    ax[5].imshow(clamp_and_detach(target_silhouette))
    for ax_, title_ in zip(
        ax,
        (
            "loss color", "rendered image", "rendered silhouette",
            "loss silhouette", "target image",  "target silhouette",
        )
    ):
        if not title_.startswith('loss'):
            ax_.grid("off")
            ax_.axis("off")
        ax_.set_title(title_)
    fig.canvas.draw(); fig.show()
    display.clear_output(wait=True)
    display.display(fig)
    return fig


# ## 5. Fit the radiance field
# 
# Here we carry out the radiance field fitting with differentiable rendering.
# 
# In order to fit the radiance field, we render it from the viewpoints of the `target_cameras`
# and compare the resulting renders with the observed `target_images` and `target_silhouettes`.
# 
# The comparison is done by evaluating the mean huber (smooth-l1) error between corresponding
# pairs of `target_images`/`rendered_images` and `target_silhouettes`/`rendered_silhouettes`.
# 
# Since we use the `MonteCarloRaysampler`, the outputs of the training renderer `renderer_mc`
# are colors of pixels that are randomly sampled from the image plane, not a lattice of pixels forming
# a valid image. Thus, in order to compare the rendered colors with the ground truth, we 
# utilize the random MonteCarlo pixel locations to sample the ground truth images/silhouettes
# `target_silhouettes`/`rendered_silhouettes` at the xy locations corresponding to the render
# locations. This is done with the helper function `sample_images_at_mc_locs`, which is
# described in the previous cell.

# In[ ]:
# render_size describes the size of both sides of the 
# rendered images in pixels. Since an advantage of 
# Neural Radiance Fields are high quality renders
# with a significant amount of details, we render
# the implicit function at double the size of 
# target images.

render_size_height = next(iter(images_data_loader))[0].shape[1]
render_size_width = next(iter(images_data_loader))[0].shape[2]

# Our rendered scene is centered around (0,0,0) 
# and is enclosed inside a bounding box
# whose side is roughly equal to 3.0 (world units).
volume_extent_world = 3.0

# 1) Instantiate the raysamplers.

# Here, NDCMultinomialRaysampler generates a rectangular image
# grid of rays whose coordinates follow the PyTorch3D
# coordinate conventions.
raysampler_grid = NDCMultinomialRaysampler(
    image_height=render_size_height,
    image_width=render_size_width,
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


# First move all relevant variables to the correct device.
renderer_grid = renderer_grid.to(device)
renderer_mc = renderer_mc.to(device)

# Set the seed for reproducibility
torch.manual_seed(1)

# Instantiate the radiance field model.
neural_radiance_field = NeuralRadianceField().to(device)

# Instantiate the Adam optimizer. We set its master learning rate to 1e-3.
lr = 1e-3
optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr)

# 3000 iterations take ~20 min on a Tesla M40 and lead to
# reasonably sharp results. However, for the best possible
# results, we recommend setting n_iter=20000.
n_epochs = 15

# Init the loss history buffers.
loss_history_color, loss_history_sil = [], []
R, T = look_at_view_transform(dist=2, elev=84, azim=-180)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)

nerf = NeuralRadianceField()

# The main optimization loop.
for epochs in range(n_epochs):      
    
    for images, configurations in images_data_loader:
        # Zero the optimizer gradient.
        optimizer.zero_grad()

        rendered_pixels, sampled_rays = renderer_mc(
            cameras=cameras,
            volumetric_function=nerf,
            configuration=configurations
        )

        ground_thruth_pixels = sample_images_at_mc_locs(images, sampled_rays.xys)

        loss = (rendered_pixels - ground_thruth_pixels).abs().mean()
        
        print(f"Loss is: {loss}")
        # Take the optimization step.
        loss.backward()
        optimizer.step()
    


# ## 6. Visualizing the optimized neural radiance field
# 
# Finally, we visualize the neural radiance field by rendering from multiple viewpoints that rotate around the volume's y-axis.

# In[ ]:



# ## 7. Conclusion
# 
# In this tutorial, we have shown how to optimize an implicit representation of a scene such that the renders of the scene from known viewpoints match the observed images for each viewpoint. The rendering was carried out using the PyTorch3D's implicit function renderer composed of either a `MonteCarloRaysampler` or `NDCMultinomialRaysampler`, and an `EmissionAbsorptionRaymarcher`.
