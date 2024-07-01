import argparse
import os
import sys
import torch
import math
import torch.nn as nn
from pytorch3d.transforms import euler_angles_to_matrix
from GenerateRobotConfigurations import generate_robot_configurations
from GenerateRobotImages import generate_3d_images
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.abspath(".") + "/src"
sys.path.append(base_dir)
device="cuda:1"

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', nargs='?', const=base_dir, type=str, default=base_dir)
parser.add_argument('--width', nargs='?', const=200, type=int, default=200)
parser.add_argument('--height', nargs='?', const=200, type=int, default=200)
parser.add_argument('--urdf_file', nargs='?', const=os.path.join(base_dir, "urdfs/Panda/panda.urdf"), type=str, default=os.path.join(base_dir, "urdfs/Panda/panda.urdf"))
parser.add_argument('--dataset', nargs='?', const=os.path.join(base_dir, "generated-dataset/ee-configurations.pt"), type=str, default=os.path.join(base_dir, "generated-dataset/ee-configurations.pt"))
parser.add_argument('--images_folder', nargs='?', const=os.path.join(base_dir, "generated-images"), type=str, default=os.path.join(base_dir, "generated-images"))
args = parser.parse_args()

generate_3d_images(args, np.array([ 0.0200, -0.9641, -0.0662, -2.7979, -0.0469,  1.9289,  0.9137]))