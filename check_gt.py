import open3d as o3d
import copy
import numpy as np
from utils.visualization.pcd_visualizer_new import draw_registration_result


def calculate_error_weighted(cloud1: o3d.geometry.PointCloud, cloud2: o3d.geometry.PointCloud) -> float:
    assert len(cloud1.points) == len(cloud2.points), "len(cloud1.points) != len(cloud2.points)"

    centroid, _ = cloud1.compute_mean_and_covariance()
    weights = np.linalg.norm(np.asarray(cloud1.points) - centroid, 2, axis=1)
    distances = np.linalg.norm(np.asarray(cloud1.points) - np.asarray(cloud2.points), 2, axis=1) / len(weights)
    return np.sum(distances / weights).item()

def calculate_error_plain(cloud1: o3d.geometry.PointCloud, cloud2: o3d.geometry.PointCloud) -> float:
    assert len(cloud1.points) == len(cloud2.points), "len(cloud1.points) != len(cloud2.points)"

    centroid, _ = cloud1.compute_mean_and_covariance()
    distances = np.linalg.norm(np.asarray(cloud1.points) - np.asarray(cloud2.points), 2, axis=1)
    return np.mean(distances).item()


if __name__ == '__main__':
    pcd_file = "./pcd_export/reef_underwater_0.02.pcd"
    pcd_gt = o3d.io.read_point_cloud(pcd_file)

    pcd_aligned = copy.deepcopy(pcd_gt)

    gt_transform_file = "./results/registration_manual_icp.txt"
    gt_transform = np.loadtxt(gt_transform_file)
    pcd_gt = pcd_gt.transform(gt_transform)

    # aligned_file = "./results/reef_feature=3Dsmoothnet_config=TEASER.txt"
    # aligned_file = "./results/reef_feature=3Dsmoothnet_config=RANSAC.txt"
    # aligned_file = "./results/reef_feature=FPFH_voxel_size=0.1config=TEASER.txt"
    aligned_file = "./results/reef_feature=FPFH_voxel_size=0.1config=RANSAC.txt"

    aligned_transform = np.loadtxt(aligned_file)
    pcd_aligned = pcd_aligned.transform(aligned_transform)

    error_weighted = calculate_error_weighted(pcd_gt, pcd_aligned)
    error_plain = calculate_error_plain(pcd_gt, pcd_aligned)

    print("Error weighted: ", error_weighted)
    print("Error plain: ", error_plain)

    gt_tmp = o3d.t.geometry.PointCloud().from_legacy(pcd_gt)
    aligned_tmp = o3d.t.geometry.PointCloud().from_legacy(pcd_aligned)

    draw_registration_result(gt_tmp, aligned_tmp,
                             np.eye(4),
                             source_attribute='uniform',
                             target_attribute='uniform')

    target_tmp = o3d.t.io.read_point_cloud("pcd_export/reef_drone_local.pcd")

    draw_registration_result(target_tmp, aligned_tmp, np.eye(4), source_attribute='uniform',
                             target_attribute='uniform')

