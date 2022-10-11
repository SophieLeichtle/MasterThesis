from fileinput import filename
import logging
import os
from sys import platform
import cv2
import matplotlib.pyplot as plt

import numpy as np
import yaml

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.external.pybullet_tools.utils import quat_from_euler
from igibson.objects.articulated_object import URDFObject
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.robots.turtlebot import Turtlebot
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.utils import let_user_pick, parse_config

def main(selection="user", headless=False, short_exec=False):
    
    config = parse_config(os.path.join(igibson.configs_path, "turtlebot_static_nav.yaml"))
    settings = MeshRendererSettings(enable_shadow=False, msaa=False,)
    s = Simulator(
        mode="gui_non_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )
    scene = EmptyScene(render_floor_plane=False,floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    # scene.load_object_categories(benchmark_names)
    s.import_scene(scene)  

    print(s.scene.get_objects())

    category = "folding_chair"
    model = "folding_chair_0019"
    
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


    camera_pose = np.array([-1, 0, 1.2])
    view_direction = np.array([1, 0, -1])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    s.renderer.set_fov(90)
    frames = s.renderer.render(modes=("rgb", "seg"))

    seg = (frames[1][:,:,0:1]*512).astype(np.int32)
    frames[1][:,:,:3] = seg
    
    plt.figure()
    plt.imshow(seg)
    plt.show()

    frames = cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR)
    cv2.imshow("image", frames)
    cv2.waitKey(0)

    for i in range(100000):
        s.step()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()