import laspy
import numpy as np
import open3d as o3d


def las2pcd(las: laspy.LasData, color: bool = False, intensity: bool = False,
            use_scale: bool = True) -> o3d.t.geometry.PointCloud:
    """
    Takes a las pointcloud as input and returns the converted pcd
    @param las: input pointcloud
    @param color: add color to the returned pcd
    @param intensity: add intensity to the returned pcd
    @param use_scale: use the scale present in the las file to scale the XYZ coordinates
    @return: converted pcd
    """
    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)

    if use_scale:
        scales = las.header.scales
        las_points = np.stack([las.X * scales[0], las.Y * scales[1], las.Z * scales[2]], axis=0).transpose((1, 0))
    else:
        las_points = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))

    pcd.point.positions = o3d.core.Tensor(las_points, dtype, device)

    if color:
        # rgb are stored as 16-bit unsigned integer, we want a float between 0 and 1
        las_colors = np.stack([las.red, las.green, las.blue],
                              axis=0).transpose((1, 0)) / (2 ** 16 - 1)
        pcd.point.colors = o3d.core.Tensor(las_colors, dtype, device)

    if intensity:
        las_intensity = np.stack([las.intensity],
                                 axis=0).transpose((1, 0)) / (2 ** 16 - 1)
        pcd.point.intensity = o3d.core.Tensor(las_intensity, dtype, device)

    return pcd
