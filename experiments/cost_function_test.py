import time
import numpy as np
import pybullet as p
import argparse
import os
import open3d as o3d

from igibson.external.pybullet_tools.utils import quat_from_euler

from igibson.simulator import Simulator
from igibson.scenes.empty_scene import EmptyScene
from igibson.robots import REGISTERED_ROBOTS
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_model_path
from igibson.objects.articulated_object import URDFObject
from igibson.utils.semantics_utils import get_class_name_to_class_id

import soph
from soph.utils.utils import bbox, openglf_to_wf
from soph.utils.pcd_dict import PointCloudDict
from soph.utils.partial_chamfer_dist import partial_chamfer_dist

ARROWS = {
    0: "up_arrow",
    1: "down_arrow",
    2: "left_arrow",
    3: "right_arrow",
    65295: "left_arrow",
    65296: "right_arrow",
    65297: "up_arrow",
    65298: "down_arrow",
}

gui = "ig"

class KeyboardController:

    def __init__(self, simulator):
        self.simulator = simulator
        self.last_keypress = None
        self.keypress_mapping = None
        self.populate_keypress_mapping()
        self.time_last_keyboard_input = time.time()
        self.picture_flag = 0

    def populate_keypress_mapping(self):
        self.keypress_mapping = {}

        self.keypress_mapping["i"] = {"idx": 0, "val": 0.2}
        self.keypress_mapping["k"] = {"idx": 0, "val": -0.2}
        self.keypress_mapping["l"] = {"idx": 1, "val": 0.1}
        self.keypress_mapping["j"] = {"idx": 1, "val": -0.1}

    def get_teleop_action(self):
        action = np.zeros(2)
        keypress = self.get_keyboard_input()
        self.picture_flag = 0

        if keypress is not None:
            if keypress in self.keypress_mapping:
                action_info = self.keypress_mapping[keypress]
                idx, val = action_info["idx"], action_info["val"]
                if idx is not None:
                    action[idx] = val
            elif keypress == " ":
                if self.last_keypress != keypress:
                    self.picture_flag = 1

        self.last_keypress = keypress

        return action


    def get_keyboard_input(self):
        """
        Checks for newly received user inputs and returns the first received input, if any
        :return None or str: User input in string form. Note that only the characters mentioned in
        @self.print_keyboard_teleop_info are explicitly supported
        """
        global gui

        # Getting current time
        current_time = time.time()
        if gui == "pb":
            kbe = p.getKeyboardEvents()
            # Record the first keypress if any was detected
            keypress = -1 if len(kbe.keys()) == 0 else list(kbe.keys())[0]
        else:
            # Record the last keypress if it's pressed after the last check
            keypress = (
                -1
                if self.simulator.viewer.time_last_pressed_key is None
                or self.simulator.viewer.time_last_pressed_key < self.time_last_keyboard_input
                else self.simulator.viewer.last_pressed_key
            )
        # Updating the time of the last check
        self.time_last_keyboard_input = current_time

        if keypress in ARROWS:
            # Handle special case of arrow keys, which are mapped differently between pybullet and cv2
            keypress = ARROWS[keypress]
        else:
            # Handle general case where a key was actually pressed (value > -1)
            keypress = chr(keypress) if keypress > -1 else None

        return keypress

def updatePointcloud(pointcloud, simulator, robot, goal_id):
    frames = simulator.renderer.render_robot_cameras(modes=("3d", "seg"))
    seg = (frames[1][:,:,0:1]*512).astype(np.int32)
    depth = frames[0]
    detections = np.unique(seg)
    print(detections)
    if goal_id in detections:
        rmin, rmax, cmin, cmax = bbox(seg == goal_id)      
        for r in range(rmin, rmax+1):
            for c in range(cmin, cmax + 1):
                if seg[r,c,0] != goal_id: continue
                point = depth[r,c,:]
                point_in_wf = np.dot(openglf_to_wf(robot), point)
                pointcloud.insert(point_in_wf[:3])

def main():
    """
    Main
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
    args = parser.parse_args()

    s = Simulator(
        mode="gui_interactive",
        image_width=640,
        image_height=480,
        vertical_fov=45
    )
    scene = EmptyScene(render_floor_plane=True,floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)
    robot = REGISTERED_ROBOTS["Turtlebot"](
        action_type="continuous",
        action_normalize=True,
        controller_config={
            "base": {"name": "DifferentialDriveController"}
        },
    )
    s.import_object(robot)
    robot.set_position([-0.75, 1.0, 0])
    robot.reset()
    robot.keep_still()

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
                fixed_base=True,
            )
    
    s.import_object(simulator_obj)
    simulator_obj.set_position_orientation([0,0,0.7], quat_from_euler([0,0,0]))
    print(simulator_obj.get_position())
    action_generator = KeyboardController(simulator=s)

    pointcloud = PointCloudDict(2,2)
    ground_truth_pointcloud = o3d.io.read_point_cloud(os.path.join(soph.point_clouds_path,category+"-"+model+".ply"))
    ground_truth_pointcloud.translate((0,0,0.7))
    max_steps = -1 
    step = 0
    while step != max_steps:
        action = (
            action_generator.get_teleop_action()
        )

        if action_generator.picture_flag == 1:
            updatePointcloud(pointcloud, s, robot, get_class_name_to_class_id()[category])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud.point_array())
            o3d.io.write_point_cloud(os.path.join(soph.point_clouds_path,"test.ply"), pcd)

            print(partial_chamfer_dist(pcd, ground_truth_pointcloud))


        robot.apply_action(action)
        for _ in range(10):
            s.step()
            step += 1

    s.disconnect()

if __name__ == "__main__":
    main()