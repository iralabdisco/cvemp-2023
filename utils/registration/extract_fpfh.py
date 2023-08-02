import numpy as np
import open3d as o3d


def pcd2xyz(pcd):
    return np.asarray(pcd.points)


def extract_features(pcd, voxel_size):
    o3d.utility.set_verbosity_level(o3d.utility.Error)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, np.array(pcd_fpfh.data).T


def extract_and_save_npz(pcd, voxel_size, path):
    pcd_down, fpfh = extract_features(pcd, voxel_size)
    pcd_xyz = pcd2xyz(pcd_down)
    np.savez_compressed(path, xyz_down=pcd_xyz, features=fpfh)
    return
