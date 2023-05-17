import numpy as np
import cv2
import os

from soph import configs_path
from soph.occupancy_grid.occupancy_grid import OccupancyGrid2D
from soph.occupancy_grid.occupancy_from_scan import get_local_occupancy_grid
from soph import DEFAULT_FOOTPRINT_RADIUS
from soph.planning.motion_planning import teleport
from igibson.utils.constants import OccupancyGridState
from scipy.ndimage import binary_erosion, binary_dilation


def spin_and_update(env, occupancy_map):
    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]
    map_cv = cv2.cvtColor(occupancy_map.grid * 255, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("pre.png", map_cv)
    for i in range(10):
        new_robot_theta = robot_theta + 0.2 * np.pi
        plan = [robot_pos[0], robot_pos[1], new_robot_theta]
        teleport(env, plan)

        state = env.get_state()
        robot_pos = env.robots[0].get_position()[:2]
        robot_theta = env.robots[0].get_rpy()[2]

        scan_grid = get_local_occupancy_grid(
            env, state["scan"], 128, 5.0, DEFAULT_FOOTPRINT_RADIUS
        )

        occupancy_map.update_with_grid_direct(
            occupancy_grid=scan_grid,
            position=robot_pos,
        )

        # Sample points from depth sensor to accompany lidar occupancy grid
        depth = state["depth"]
        occupancy_map.update_from_depth(env, depth, 5000)

        map_cv = cv2.cvtColor(scan_grid * 255, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(f"{i}.png", map_cv)
    map_cv = cv2.cvtColor(occupancy_map.grid * 255, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("post.png", map_cv)
    input("enter")


def update_grid_with_scan(env, occupancy_map):
    state = env.get_state()
    robot_pos = env.robots[0].get_position()[:2]

    scan_grid = get_local_occupancy_grid(
        env, state["scan"], 128, 5.0, DEFAULT_FOOTPRINT_RADIUS
    )

    occupancy_map.update_with_grid_direct(
        occupancy_grid=scan_grid,
        position=robot_pos,
    )


def refine_map(occupancy_map):
    free = occupancy_map.grid == OccupancyGridState.FREESPACE
    free_refined = binary_dilation(binary_erosion(free))
    outline = free ^ free_refined

    occupancy_map.grid[outline] = OccupancyGridState.UNKNOWN


def initial_position(area, gridsize=350):
    filename = os.path.join(configs_path, area + ".png")
    mapread = cv2.imread(filename)
    mapprocessed = np.round(cv2.cvtColor(mapread, cv2.COLOR_BGR2GRAY) / 255 * 2) / 2
    grid = OccupancyGrid2D(gridsize)
    grid.grid = mapprocessed

    while True:
        sample = grid.sample_uniform()
        if grid.check_if_free(sample, DEFAULT_FOOTPRINT_RADIUS * 1.25):
            break

    return [sample[0], sample[1], 0]
