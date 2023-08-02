import os

import laspy
import numpy as np
import open3d as o3d
from utils.pointcloud.las_conversion import las2pcd
from utils.registration.extract_fpfh import extract_and_save_npz
from utils.registration.registration_from_features import register
from utils.visualization.pcd_visualizer_new import draw_registration_result

if __name__ == '__main__':

    # parameters
    test_name = 'reef'
    source_name = 'reef_underwater_medium'
    target_name = 'reef_drone_local'
    convert_from_laz = False
    voxel_size = 0.1
    config = 'TEASER'
    feature_type = "FPFH"

    if feature_type == "FPFH":
        test_name = test_name + "_feature=" + feature_type + '_voxel_size=' + str(voxel_size) + 'config=' + config
    else:
        test_name = test_name + "_feature=" + feature_type + '_config=' + config

    if convert_from_laz:
        print("Reading las files")
        las_source = laspy.read('./laz_data/' + source_name + '.laz')
        las_target = laspy.read('./laz_data/' + target_name + '.laz')

        print("Las source: {}".format(las_source))
        print("Las target: {}".format(las_target))

        print("Converting to pcd")
        pcd_source = las2pcd(las_source, use_scale=True, color=True)
        pcd_target = las2pcd(las_target, use_scale=True, color=True)

        print("Write pcd to file")
        o3d.t.io.write_point_cloud('./pcd_export/' + source_name + '.pcd', pcd_source)
        o3d.t.io.write_point_cloud('./pcd_export/' + target_name + '.pcd', pcd_target)
    else:
        print("Reading source pcd")
        pcd_source = o3d.t.io.read_point_cloud('./pcd_export/' + source_name + '.pcd')
        print("Reading target pcd")
        pcd_target = o3d.t.io.read_point_cloud('./pcd_export/' + target_name + '.pcd')

    if feature_type == "FPFH":
        source_features = './npz_feature/' + source_name + '_fpfh_voxel_size=' + str(voxel_size) + '.npz'
        if not os.path.exists(source_features):
            print("Extracting features from source")
            extract_and_save_npz(pcd_source.to_legacy(), voxel_size, source_features)
        target_features = './npz_feature/' + target_name + '_fpfh_voxel_size=' + str(voxel_size) + '.npz'
        if not os.path.exists(target_features):
            print("Extracting features from target")
            extract_and_save_npz(pcd_target.to_legacy(), voxel_size, target_features)
    elif feature_type == "3Dsmoothnet":
        source_features = './npz_feature/' + source_name + "_3dsmoothnet.npz"
        target_features = './npz_feature/' + target_name + "_3dsmoothnet.npz"
    else:
        print(f"Feature type not recognized: {feature_type}")
        exit()

    registration_solution = np.eye(4)
    if config == "TEASER":
        print("Registering using TEASER++")
        registration_solution = register(source_features, target_features,
                                         config_path='./configs/' + config + '.json', n_keypoints=10000,
                                         algorithm='TEASER')
    elif config == "RANSAC":
        print("Registering using RANSAC")
        registration_solution = register(source_features, target_features,
                                         config_path='./configs/' + config + '.json', n_keypoints=10000,
                                         algorithm='RANSAC')
    print(registration_solution)

    np.savetxt('./results/' + test_name + '.txt', registration_solution)

    pcd_source_tmp = pcd_source.random_down_sample(0.1)
    pcd_target_tmp = pcd_target.random_down_sample(0.5)

    draw_registration_result(pcd_source_tmp, pcd_target_tmp,
                             registration_solution,
                             source_attribute='colors',
                             target_attribute='colors')
