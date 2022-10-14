import open3d as o3d
import argparse
import os
from soph import point_clouds_path

def main(selection="user", headless=False, short_exec=False):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        default="folding_chair-folding_chair_0019.ply",
        help="Name of the pointcloud file, assumes it is located in the folder ground_truth_pointclouds",
    )
    args = parser.parse_args()

    #pcd_load = o3d.io.read_point_cloud(os.path.join(point_clouds_path, args.path))
    pcd_load = o3d.io.read_point_cloud(os.path.join(point_clouds_path, "test.ply"))
    o3d.visualization.draw_geometries([o3d.geometry.TriangleMesh.create_coordinate_frame(), pcd_load])

if __name__ == "__main__":
    main()