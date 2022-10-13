import logging
import os
from sys import platform
import numpy as np
import open3d as o3d
import argparse


import igibson
from igibson.simulator import Simulator
from igibson.scenes.empty_scene import EmptyScene
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_model_path
from igibson.objects.articulated_object import URDFObject
from igibson.utils.semantics_utils import get_class_name_to_class_id

from utils.utils import bbox
from utils.pcd_dict import PointCloudDict

def main(selection="user", headless=False, short_exec=False):
    """
    Create a PointCloud
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        "-c",
        default="folding_chair",
        help="Which Category the object belongs to",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="folding_chair_0019",
        help="name of the object file",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=1.0,
        help="radius at which to observe the object",
        type=np.float64
    )
    parser.add_argument(
        "--iterations",
        "-i",
        default=100,
        help="number of sampled viewpoints",
        type=np.int64
    )
    args = parser.parse_args()

    s = Simulator(
        mode="gui_non_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
    )
    scene = EmptyScene(render_floor_plane=False,floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    # scene.load_object_categories(benchmark_names)
    s.import_scene(scene)  

    category = args.category
    model = args.model
    
    model_path = get_ig_model_path(category, model)
    filename = os.path.join(model_path, model + ".urdf")

    avg_category_spec = get_ig_avg_category_specs()

    obj_name = "{}_{}".format("floor_lamp", 1)
    simulator_obj = URDFObject(
                filename,
                name=obj_name,
                category=category,
                model_path=model_path,
                avg_obj_dims=avg_category_spec.get(category),
                fit_avg_dim_volume=True,
                texture_randomization=False,
                overwrite_inertial=True,
            )
    
    s.import_object(simulator_obj)

    point_cloud = PointCloudDict(precision=3, sub_precision=1)
    goal_id = get_class_name_to_class_id()[category]
    num_points = args.iterations
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)
    for i in range(0, num_points):
        x = args.radius * np.sin(theta[i]) * np.cos(phi[i])
        y = args.radius * np.sin(theta[i]) * np.sin(phi[i])
        z = args.radius * np.cos(theta[i])

        camera_pose = np.array([x,y,z])
        view_direction = np.array([-x,-y,-z])
        s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        s.renderer.set_fov(90)

        frames = s.renderer.render(modes=("3d", "seg"))
        seg = (frames[1][:,:,0:1]*512).astype(np.int32)
        depth = frames[0]
        detections = np.unique(seg)
        if goal_id in detections:
            rmin, rmax, cmin, cmax = bbox(seg == goal_id)      
            for r in range(rmin, rmax+1):
                for c in range(cmin, cmax + 1):
                    if seg[r,c,0] != goal_id: continue
                    point = depth[r,c,:]
                    point_in_wf = np.dot(np.linalg.inv(s.renderer.V), point)
                    point_cloud.insert(point_in_wf[:3])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.point_array())
    o3d.io.write_point_cloud("ground_truth_pointclouds/"+ category + "-" + model + ".ply", pcd)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()