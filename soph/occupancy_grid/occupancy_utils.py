import numpy as np

from soph.occupancy_grid.occupancy_from_scan import get_local_occupancy_grid
from soph import DEFAULT_FOOTPRINT_RADIUS
from soph.utils.motion_planning import teleport
from igibson.utils.constants import OccupancyGridState
from scipy.ndimage import binary_erosion, binary_dilation


def spin_and_update(env, occupancy_map):
    robot_pos = env.robots[0].get_position()[:2]
    robot_theta = env.robots[0].get_rpy()[2]
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
