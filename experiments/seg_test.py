import time
from tkinter import S
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import yaml
import cv2

from igibson.simulator import Simulator
from igibson.scenes.empty_scene import EmptyScene
from igibson.robots import REGISTERED_ROBOTS
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

from soph.utils.utils import bbox, openglf_to_wf

from yolo_utils import create_model, get_predictions, prepare_image, save_seg_image

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

    rendering_settings = MeshRendererSettings(
                enable_shadow=True,
                enable_pbr=True,
                msaa=True,
                texture_scale=1.0,
                optimized=True,
                load_textures=True,
                hide_robot=True
            )
    s = Simulator(
        mode="gui_interactive",
        image_width=640,
        image_height=640,
        vertical_fov=45,
        rendering_settings=rendering_settings
    )
    config_filename = "soph/configs/test_config.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    scene = EmptyScene(render_floor_plane=True,floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    scene = InteractiveIndoorScene(
        config["scene_id"],
        urdf_file=config.get("urdf_file", None),
        waypoint_resolution=config.get("waypoint_resolution", 0.2),
        num_waypoints=config.get("num_waypoints", 10),
        build_graph=config.get("build_graph", False),
        trav_map_resolution=config.get("trav_map_resolution", 0.1),
        trav_map_erosion=config.get("trav_map_erosion", 2),
        trav_map_type=config.get("trav_map_type", "with_obj"),
        texture_randomization=False,
        object_randomization=False,
        object_randomization_idx=None,
        should_open_all_doors=config.get("should_open_all_doors", False),
        load_object_categories=config.get("load_object_categories", None),
        not_load_object_categories=config.get("not_load_object_categories", None),
        load_room_types=config.get("load_room_types", None),
        load_room_instances=config.get("load_room_instances", None),
        merge_fixed_links=config.get("merge_fixed_links", True)
        and not config.get("online_sampling", False),
        include_robots=config.get("include_robots", True),
    )
    s.import_scene(scene)
    robot = REGISTERED_ROBOTS["Turtlebot"](
        action_type="continuous",
        action_normalize=True,
        controller_config={
            "base": {"name": "DifferentialDriveController"}
        },
    )
    s.import_object(robot)
    robot.set_position([0, 0, 0])
    robot.reset()
    robot.keep_still()

    action_generator = KeyboardController(simulator=s)
    
    model, device, names, colors = create_model("experiments/yolov7.pt", 640, True)

    max_steps = -1 
    step = 0
    while step != max_steps:
        action = (
            action_generator.get_teleop_action()
        )

        if action_generator.picture_flag == 1:
            [img] = s.renderer.render_robot_cameras(modes=("rgb"))
            img = np.clip(img*255, 0, 255)
            img = img.astype(np.uint8)
            img_tensor, brg_img = prepare_image(img[:,:,:3], device)
            predictions = get_predictions(img_tensor, model)
            save_seg_image("seg.png", predictions, img_tensor, brg_img, names, colors)
            #cv2.imwrite("seg.png", img_with_seg)

        robot.apply_action(action)
        for _ in range(10):
            s.step()
            step += 1

    s.disconnect()

if __name__ == "__main__":
    main()