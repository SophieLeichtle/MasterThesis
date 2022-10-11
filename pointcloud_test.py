import open3d as o3d

pcd_load = o3d.io.read_point_cloud("testocto.ply")
print(pcd_load)
o3d.visualization.draw_geometries([pcd_load])