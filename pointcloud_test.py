import open3d as o3d

pcd_load = o3d.io.read_point_cloud("ground_truth_pointclouds/fridge-10905.ply")
print(pcd_load)
o3d.visualization.draw_geometries([pcd_load])