import numpy as np
from transforms3d.quaternions import quat2mat
from igibson.utils.constants import OccupancyGridState
import cv2


def get_local_occupancy_grid(
    env, scan, grid_resolution, occupancy_range, robot_footprint_radius, rear=False
):
    """
    Get local occupancy grid based on current 1D scan

    :param: 1D LiDAR scan
    :return: local occupancy grid
    """
    laser_linear_range = env.config.get("laser_linear_range", 10.0)
    laser_angular_range = env.config.get("laser_angular_range", 180.0)
    min_laser_dist = env.config.get("min_laser_dist", 0.05)
    n_horizontal_rays = env.config.get("n_horizontal_rays", 128)
    laser_link_name = (
        env.config.get("laser_link_name", "scan_link")
        if not rear
        else env.config.get("laser_link_rear_name", "scan_link")
    )
    laser_position, laser_orientation = (
        env.robots[0].links[laser_link_name].get_position_orientation()
    )

    laser_angular_half_range = laser_angular_range / 2.0

    angle = np.arange(
        -np.radians(laser_angular_half_range),
        np.radians(laser_angular_half_range),
        np.radians(laser_angular_range) / n_horizontal_rays,
    )
    unit_vector_laser = np.array([[np.cos(ang), np.sin(ang), 0.0] for ang in angle])

    scan_laser = unit_vector_laser * (
        scan * (laser_linear_range - min_laser_dist) + min_laser_dist
    )

    laser_translation = laser_position
    laser_rotation = quat2mat(
        [
            laser_orientation[3],
            laser_orientation[0],
            laser_orientation[1],
            laser_orientation[2],
        ]
    )
    scan_world = laser_rotation.dot(scan_laser.T).T + laser_translation

    base_position, base_orientation = env.robots[0].base_link.get_position_orientation()
    base_rotation = np.eye(3)
    scan_local = base_rotation.T.dot((scan_world - base_position).T).T
    scan_local = scan_local[:, :2]
    scan_local = np.concatenate(
        [np.array([[0, 0]]), scan_local, np.array([[0, 0]])], axis=0
    )

    # flip y axis
    scan_local[:, 1] *= -1

    occupancy_grid = np.zeros((grid_resolution, grid_resolution)).astype(np.uint8)

    occupancy_grid.fill(int(OccupancyGridState.UNKNOWN * 2.0))
    scan_local_in_map = scan_local / occupancy_range * grid_resolution + (
        grid_resolution / 2
    )
    scan_local_in_map = scan_local_in_map.reshape((1, -1, 1, 2)).astype(np.int32)

    cv2.fillPoly(
        img=occupancy_grid,
        pts=scan_local_in_map,
        color=int(OccupancyGridState.FREESPACE * 2.0),
        lineType=1,
    )

    for i in range(1, scan_local_in_map.shape[1] - 2):
        if scan[i, 0] >= 0.95:
            continue
        cv2.rectangle(
            img=occupancy_grid,
            pt1=(scan_local_in_map[0, i, 0, 0], scan_local_in_map[0, i, 0, 1]),
            pt2=(scan_local_in_map[0, i, 0, 0] + 1, scan_local_in_map[0, i, 0, 1] + 1),
            color=int(OccupancyGridState.OBSTACLES),
            thickness=-1,
        )

    robot_footprint_radius_in_map = int(
        robot_footprint_radius / occupancy_range * grid_resolution
    )
    cv2.circle(
        img=occupancy_grid,
        center=(grid_resolution // 2, grid_resolution // 2),
        radius=int(robot_footprint_radius_in_map),
        color=int(OccupancyGridState.FREESPACE * 2.0),
        thickness=-1,
    )

    return occupancy_grid[:, :, None].astype(np.float32) / 2.0
