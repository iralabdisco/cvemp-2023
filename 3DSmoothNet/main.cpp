/**
3DSmoothNet
main.cpp

Purpose: executes the computation of the SDV voxel grid for the selected interes points

@Author : Zan Gojcic, Caifa Zhou
@Version : 1.0
*/

#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include "core/core.h"


int main(int argc, char *argv[])
{
    // Turn off the warnings of pcl
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    // Initialize the variables
    std::string data_file;
    float radius;
    int num_voxels;
    float smoothing_kernel_width;
    float perc_points = 0.01;
    std::string output_folder;

    // Get command line arguments
    bool result = processCommandLine(argc, argv, data_file, radius, num_voxels, smoothing_kernel_width, perc_points, output_folder);
    if (!result)
        return 1;

    // Check if the output folder exists, otherwise create it
    boost::filesystem::path dir(output_folder);
    if (boost::filesystem::create_directory(dir))
    {
        std::cerr << "Directory Created: " << output_folder << std::endl;
    }

    // Read in the point cloud using the ply reader
    // std::cout << "Config parameters successfully read in!! \n" << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (fileExist(data_file))
        pcl::io::loadPCDFile(data_file, *cloud);
    else
    {
        std::cout << "Point cloud file does not exsist or cannot be opened!!" << std::endl;
        return 1;
    }
    std::vector<int> nans;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, nans);

    // std::cout << "File: " << data_file << std::endl;
    // std::cout << "Number of Points: " << cloud->size() << std::endl;
    // std::cout << "Size of the voxel grid: " << 2 * radius << std::endl; // Multiplied with two as half size is used (corresponding to the radius)
    // std::cout << "Number of Voxels: " << num_voxels << std::endl;
    // std::cout << "Smoothing Kernel: " << smoothing_kernel_width << std::endl;



    // Specify the parameters of the algorithm
    const int grid_size = num_voxels * num_voxels * num_voxels;
    float voxel_step_size = (2 * radius) / num_voxels;
    float lrf_radius = sqrt(3)*radius; // Such that the circumscribed sphere is obtained

    // Initialize the voxel grid
    flann::Matrix<float> voxel_coordinates = initializeGridMatrix(num_voxels, voxel_step_size, voxel_step_size, voxel_step_size);

    // Compute the local reference frame for all the points
    float smoothing_factor = smoothing_kernel_width * (radius / num_voxels); // Equals half a voxel size so that 3X is 1.5 voxel

    // Check if all the points should be evaluated or only selected  ones
    std::string flag_all_points = "0";
    std::vector<int> evaluation_points;

    pcl::RandomSample <pcl::PointXYZ> random_sampling;
    random_sampling.setInputCloud(cloud);
    random_sampling.setSeed (std::rand ());
    random_sampling.setSample(perc_points);
    random_sampling.filter(evaluation_points);


    std::size_t found = data_file.find_last_of("/");
    std::string temp_token = data_file.substr(found + 1);
    std::size_t found2 = temp_token.find_last_of(".");

    std::string save_file_name = temp_token.substr(0, found2);
    save_file_name = output_folder + save_file_name;

    //Saving keypoints
    std::cout << "Keypoints path: " << save_file_name;
    pcl::io::savePCDFile(save_file_name+"_keypoints.pcd", *cloud, evaluation_points, false);

    //std::cout << "Number of keypoints:" << evaluation_points.size() << "\n" << std::endl;


    // Initialize the variables for the NN search and LRF computation
    std::vector<int> indices(cloud->width);
    std::vector<LRF> cloud_lrf(cloud->width);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector <std::vector <int>> nearest_neighbors(cloud->width);
    std::vector <std::vector <int>> nearest_neighbors_smoothing(cloud->width);
    std::vector <std::vector <float>> nearest_neighbors_smoothing_dist(cloud->width);

    // Compute the local reference frame for the interes points (code adopted from https://www.researchgate.net/publication/310815969_TOLDI_An_effective_and_robust_approach_for_3D_local_shape_description
    // and not optimized)

    auto t1_lrf = std::chrono::high_resolution_clock::now();
    toldiComputeLRF(cloud, evaluation_points, lrf_radius, 3 * smoothing_factor, cloud_lrf, nearest_neighbors, nearest_neighbors_smoothing, nearest_neighbors_smoothing_dist);
    auto t2_lrf = std::chrono::high_resolution_clock::now();

    // Compute the SDV representation for all the points


    // Start the actuall computation
    auto t1 = std::chrono::high_resolution_clock::now();
    computeLocalDepthFeature(cloud, evaluation_points, nearest_neighbors, cloud_lrf, radius, voxel_coordinates, num_voxels, smoothing_factor, save_file_name);
    auto t2 = std::chrono::high_resolution_clock::now();
     std::cout << "\n---------------------------------------------------------" << std::endl;
     std::cout << "LRF computation took "
               << std::chrono::duration_cast<std::chrono::milliseconds>(t2_lrf - t1_lrf).count()
               << " miliseconds\n";
     std::cout << "SDV computation took "
               << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
               << " miliseconds\n";
     std::cout << "---------------------------------------------------------" << std::endl;


    return 0;

}
