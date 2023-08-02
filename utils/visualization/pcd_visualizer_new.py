import copy

import laspy
import matplotlib
import numpy as np
import open3d as o3d

import pcr_utils.pointcloud.las_conversion


def color_pcd(pcd: o3d.t.geometry.PointCloud, attribute: str = 'uniform',
              color_rgb=None):
    if attribute == 'uniform' and color_rgb is None:
        color_rgb = [1, 0.706, 0]

    if attribute == 'uniform':
        colors_to_paint = np.tile(np.array(color_rgb), (len(pcd.point.positions), 1)).astype(
            'float32')
        pcd.point.colors = colors_to_paint
        return pcd

    if attribute == 'colors':
        print("Using default colors")
        return pcd

    if attribute in pcd.point:
        attribute_values = pcd.point[attribute].numpy().flatten()
        attribute_values = attribute_values / max(attribute_values)
        colormap = matplotlib.colormaps.get_cmap('hot')
        colors_to_paint = colormap(attribute_values)
        pcd.point.colors = np.array(colors_to_paint[:, :3]).astype('float32')
        return pcd
    else:
        raise Exception("Attribute {} not present in pcd".format(attribute))


def visualize_pcd(pcd: o3d.t.geometry.PointCloud, voxel_size: float = 1,
                  attribute: str = 'uniform', color_rgb=None):

    if attribute == 'uniform' and color_rgb is None:
        color_rgb = [1, 0.706, 0]

    pcd_temp = copy.deepcopy(pcd)
    pcd_temp = pcd_temp.voxel_down_sample(voxel_size=voxel_size)
    pcd_temp = color_pcd(pcd_temp, attribute, color_rgb)

    o3d.visualization.draw([pcd_temp], point_size=1, bg_color=[0, 0, 0, 1], raw_mode=True, show_skybox=False,
                           show_ui=True)
    return


def draw_registration_result(source: o3d.t.geometry.PointCloud, target: o3d.t.geometry.PointCloud,
                             transformation: np.ndarray, voxel_size: float = -1,
                             source_attribute: str = 'uniform',
                             target_attribute: str = 'uniform'):

    # Workaround because source_temp = copy.deepcopy(source) still modifies the original pcd
    source_temp = o3d.t.geometry.PointCloud()
    for attribute in source.point:
        source_temp.point[attribute] = copy.deepcopy(source.point[attribute])
    target_temp = o3d.t.geometry.PointCloud()
    for attribute in target.point:
        target_temp.point[attribute] = copy.deepcopy(target.point[attribute])

    if voxel_size > 0:
        source_temp = source_temp.voxel_down_sample(voxel_size)
        target_temp = target_temp.voxel_down_sample(voxel_size)

    source_temp = source_temp.transform(transformation)

    if source_attribute == 'uniform':
        source_temp = color_pcd(source_temp, attribute='uniform', color_rgb=[1.0, 0.706, 0])
    else:
        source_temp = color_pcd(source_temp, attribute=source_attribute)

    if target_attribute == 'uniform':
        target_temp = color_pcd(target_temp, attribute='uniform', color_rgb=[0.0, 0.651, 0.929])
    else:
        target_temp = color_pcd(target_temp, attribute=target_attribute)

    o3d.visualization.draw([source_temp, target_temp], point_size=1, bg_color=[0, 0, 0, 1], raw_mode=True,
                           show_skybox=False,
                           show_ui=True)
    return


if __name__ == '__main__':
    las = laspy.read('/home/fdila/Downloads/pcr_luca/LAS_tests/falesia_lidar.laz')
    print(las.header.scales)
    print(las.X[:10])
    print(las.x[:10])
    pcd_from_las = pcr_utils.pointcloud.las_conversion.las2pcd(las, use_scale=False)
    visualize_pcd(pcd_from_las, voxel_size=1, attribute='colors')
