import matplotlib.pyplot as plt
from itertools import chain, combinations
import numpy as np

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


def create_powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generate_configurations(number_per_powerset):
	powersets = create_powerset([0,1,2,3,4,5,6])
	joint_limits = [
			(-2.8973, 2.8973),    # Joint 1
			(-1.7628, 1.7628),    # Joint 2
			(-2.8973, 2.8973), # Joint 3
			(-3.0718, -0.0698),    # Joint 4
			(-2.8973, 2.8973), # Joint 5
			(-0.0175, 3.7525),    # Joint 6
			(-2.8973, 2.8973)  # Joint 7
		]

	all_configurations = np.empty((0,7))
	for powerset in powersets:
			if powerset == ():
					continue

			# set temporary limits for current subset
			temp_limits = np.array([(0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)])
			for index in powerset:
					temp_limits[index] = joint_limits[index]

			configurations = np.array([
				np.random.uniform(low=min_angle, high=max_angle, size=number_per_powerset)
				for min_angle, max_angle in temp_limits
			]).T

			all_configurations = np.vstack([all_configurations, configurations])

	return all_configurations