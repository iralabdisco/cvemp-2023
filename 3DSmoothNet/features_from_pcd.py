#!/usr/bin/env python
# -*- coding: future_fstrings -*-

import os
import argparse
import logging

import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd
import open3d as o3d
from pathlib import Path
import yaml
from types import SimpleNamespace
import open3d
import subprocess

import benchmark_helper


def main():
    parser = argparse.ArgumentParser(description='3DSmoothNet Voxelization')
    parser.add_argument('parameters', type=str, help='Path to the yaml parameters file')
    args = parser.parse_args()

    with open(args.parameters, 'r') as file:
        parameter = yaml.load(file, Loader=yaml.FullLoader)
    print(parameter)
    params = SimpleNamespace(**parameter)

    output_dir = "/data/disat/features/"
    pcd_path = "/data/disat/pcd/"
    pcd_name = "reef_underwater_0.02"

    voxelization_parameters = f"""-k {params.num_points} -r {params.voxel_size} -n {params.vox_num} -h {params.gauss_width} -o {output_dir}"""

    os.system(f"/home/user/3DSmoothNet/3DSmoothNet -f {pcd_path+pcd_name+'.pcd'} {voxelization_parameters}")

    args = "python3 main_cnn.py --run_mode=test --evaluate_input_folder=/data/disat/features/  --evaluate_output_folder=/data/disat/features/"
    subprocess.call(args, shell=True)

    n_dim = 64
    features = np.load(output_dir + str(n_dim) + "_dim/" + pcd_name + "_3DSmoothNet.npz")['data']
    keypoints = open3d.io.read_point_cloud(os.path.join(output_dir + pcd_name + '_keypoints.pcd'))
    keypoints = np.asarray(keypoints.points)

    save_name = os.path.join(output_dir, pcd_name +".npz")

    np.savez_compressed(save_name, xyz_down=keypoints, features=features)

if __name__ == '__main__':
    main()