import argparse
import os
import sys
import torch
from GenerateRobotConfigurations import generate_robot_configurations
from GenerateRobotImages import generate_3d_images

base_dir = os.path.abspath(".") + "/src"
sys.path.append(base_dir)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', nargs='?', const=base_dir, type=str, default=base_dir)
	parser.add_argument('--width', nargs='?', const=400, type=int, default=400)
	parser.add_argument('--height', nargs='?', const=400, type=int, default=400)
	parser.add_argument('--urdf_file', nargs='?', const=os.path.join(base_dir, "urdfs/Panda/panda.urdf"), type=str, default=os.path.join(base_dir, "urdfs/Panda/panda.urdf"))
	parser.add_argument('--dataset', nargs='?', const=os.path.join(base_dir, "generated-dataset/ee-configurations.pt"), type=str, default=os.path.join(base_dir, "generated-dataset/ee-configurations.pt"))
	parser.add_argument('--images_folder', nargs='?', const=os.path.join(base_dir, "generated-images"), type=str, default=os.path.join(base_dir, "generated-images"))
	args = parser.parse_args()

	if (not os.path.exists(os.path.join(base_dir, args.dataset))):
		generate_robot_configurations(1000)

	dataset = torch.load(args.dataset)
	for (ee_pose, configuration) in dataset:
		generate_3d_images(args,configuration.numpy())