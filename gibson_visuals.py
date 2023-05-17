import os
import logging
from enum import IntEnum
import yaml
import numpy as np
import time
import csv
import argparse

from soph.yolo.yolo_mask_utils import create_model, get_detections

from soph import configs_path
from soph.environments.custom_env import CustomEnv
from soph.occupancy_grid.occupancy_grid import OccupancyGrid2D
from soph.occupancy_grid.occupancy_utils import (
    spin_and_update,
    refine_map,
    initial_position,
)

from soph.planning.rt_rrt_star.rt_rrt_star import RTRRTstar
from soph.planning.rt_rrt_star.rt_rrt_star_planning import (
    goal_from_poi,
    next_goal,
)

from soph.planning.motion_planning import teleport, get_poi, FrontierSelectionMethod
from soph.utils.logging_utils import (
    initiate_logging,
    save_map_rt_rrt_star,
)

from soph.utils.detection_tool import DetectionTool
from soph import DEFAULT_FOOTPRINT_RADIUS


def main(config="benevolence0"):
    config_filename = os.path.join(configs_path, config + ".yaml")
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Create Environment
    logging.info("Creating Environment")
    env = CustomEnv(config_file=config_data, mode="gui_interactive")
    env.reset()

    while True:
        env.step(None)


if __name__ == "__main__":
    main()
