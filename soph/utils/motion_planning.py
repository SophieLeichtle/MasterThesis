from soph.utils.utils import bbox
import numpy as np
from igibson.external.pybullet_tools.utils import plan_base_motion_2d

def plan_base_motion(
    robot,
    goal,
    map,
    visualize_planning=False,
    visualize_result=False,
    optimize_iter=0,
    algorithm="birrt",
    robot_footprint_radius=0.32
):
    """
    Plan base motion given a base goal
    :param robot: Robot with which to perform planning
    :param goal: Goal Configuration (x, y, theta)
    :param map: Occupancy Grid (of type OccupancyGrid2D)
    :param visualize_planning: boolean. To visualize the planning process
    :param visualize_result: boolean. to visualize the planning results
    :param optimize_iter: iterations of the optimizer to run after path has been found
    :param algorithm: planning algorithm to use 
    :param robot_footprint_radius: robot footprint in meters
    """
    x,y,theta = goal

    rmin, rmax, cmin, cmax = bbox(map.grid != 0.5)
    corners = (tuple(map.px_to_m(np.asarray([rmax, cmin]))),tuple(map.px_to_m(np.asarray([rmin, cmax]))))
    
    occupancy_range = (map.half_size * 2 + 1) / map.m_to_pix_ratio
    grid_resolution =  map.half_size * 2 + 1
    robot_footprint_radius_in_map = robot_footprint_radius / occupancy_range * grid_resolution

    def pos_to_map(pos):
        return map.m_to_px(pos).astype(np.int32)

    path = plan_base_motion_2d(
        robot.get_body_ids()[0],
        [x,y,theta],
        corners,
        map_2d=map.grid,
        occupancy_range= occupancy_range,
        grid_resolution=grid_resolution,
        robot_footprint_radius_in_map=robot_footprint_radius_in_map,
        resolutions=np.array([0.05, 0.05, 2 * np.pi]),
        metric2map=pos_to_map,
        upsampling_factor=2,
        optimize_iter=optimize_iter,
        algorithm=algorithm,
        visualize_planning=visualize_planning,
        visualize_result=visualize_result
    )

    return path