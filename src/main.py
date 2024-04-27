import argparse
import os
import sys

from src.robotMeshRenderer import RobotMeshRenderer

base_dir = os.path.abspath(".")
sys.path.append(base_dir)

from pathlib import Path


def setup_robot_renderer(self, mesh_files):
	# mesh_files: list of mesh files
	focal_length = [-self.args.fx, -self.args.fy]
	principal_point = [self.args.px, self.args.py]
	image_size = [self.args.height, self.args.width]

	robot_renderer = RobotMeshRenderer(
		focal_length=focal_length, principal_point=principal_point, image_size=image_size,
		robot=self.robot, mesh_files=mesh_files, device=self.device)

	return robot_renderer


def main(args):
	mesh_files = [os.path.join(args.base_dir, "urdfs/meshes/collision/link0.obj"),
				  os.path.join(args.base_dir, "urdfs/meshes/collision/link1.obj"),
				  os.path.join(args.base_dir, "urdfs/meshes/collision/link2.obj"),
				  os.path.join(args.base_dir, "urdfs/meshes/collision/link3.obj"),
				  os.path.join(args.base_dir, "urdfs/meshes/collision/link4.obj"),
				  os.path.join(args.base_dir, "urdfs/meshes/collision/link5.obj"),
				  os.path.join(args.base_dir, "urdfs/meshes/collision/link6.obj"),
				  os.path.join(args.base_dir, "urdfs/meshes/collision/link7.obj")]
	robot_renderer = setup_robot_renderer(mesh_files)

def get_args():
	parser = argparse.ArgumentParser()
	args = parser.parse_args("")

	args.data_folder = "/home/sscekic/master-thesis/src/panda-data"
	args.base_dir = "/home/sscekic/master-thesis/src"
	args.use_gpu = False
	args.trained_on_multi_gpus = False
	args.keypoint_seg_model_path = os.path.join(args.base_dir, "weights/pretrain/baxter/net.pth")
	# args.keypoint_seg_model_path = os.path.join(args.base_dir,"weights/baxter/net.pth")
	args.urdf_file = os.path.join(args.base_dir, "urdfs/panda.urdf")

	##### training parameters #####
	## args.batch_size = 6
	## args.num_workers = 6
	## args.lr = 1e-6
	## args.beta1 = 0.9
	## args.n_epoch = 500
	## args.out_dir = 'outputs/Baxter_arm/weights'
	## args.ckp_per_epoch = 10
	## args.reproj_err_scale = 1.0 / 100.0
	################################

	args.robot_name = 'Panda'  # "Panda" or "Baxter_left_arm"
	args.n_kp = 7
	args.scale = 0.3125
	args.height = 1536
	args.width = 2048
	args.fx, args.fy, args.px, args.py = 960.41357421875, 960.22314453125, 1021.7171020507812, 776.2381591796875

	# scale the camera parameters
	args.width = int(args.width * args.scale)
	args.height = int(args.height * args.scale)
	args.fx = args.fx * args.scale
	args.fy = args.fy * args.scale
	args.px = args.px * args.scale
	args.py = args.py * args.scale

	return args


if __name__ == '__main__':
	args = get_args()
	main(args)
