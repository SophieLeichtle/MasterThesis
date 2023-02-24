from __future__ import print_function

import numpy as np

from igibson.external.pybullet_tools.utils import MAX_DISTANCE, circular_difference, CIRCULAR_LIMITS, get_base_difference_fn, get_base_distance_fn

from igibson.external.motion.motion_planners.rrt_connect import birrt, direct_path
from igibson.external.motion.motion_planners.rrt_star import rrt_star
from igibson.external.motion.motion_planners.lazy_prm import lazy_prm_replan_loop
from igibson.external.motion.motion_planners.rrt import rrt
from igibson.external.motion.motion_planners.smoothing import optimize_path
from igibson.utils.constants import OccupancyGridState
#from ..motion.motion_planners.rrt_connect import birrt, direct_path
import cv2
import logging

from soph.utils.utils import bbox
from igibson.utils.constants import OccupancyGridState


from soph import DEFAULT_FOOTPRINT_RADIUS

log = logging.getLogger(__name__)

def plan_base_motion_custom(
    start,
    goal,
    map,
    visualize_planning=False,
    visualize_result=False,
    optimize_iter=0,
    algorithm="birrt",
    robot_footprint_radius=DEFAULT_FOOTPRINT_RADIUS
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
    x_start,y_start,theta_start = start
    x,y,theta = goal

    rmin, rmax, cmin, cmax = bbox(map.grid != 0.5)
    corners = (tuple(map.px_to_m(np.asarray([rmax, cmin]))),tuple(map.px_to_m(np.asarray([rmin, cmax]))))
    
    occupancy_range = (map.half_size * 2 + 1) / map.m_to_pix_ratio
    grid_resolution =  map.half_size * 2 + 1
    robot_footprint_radius_in_map = int(robot_footprint_radius / occupancy_range * grid_resolution)

    def pos_to_map(pos):
        return map.m_to_px(pos).astype(np.int32)

    path = plan_base_motion_2d_custom(
        [x_start,y_start,theta_start],
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

def plan_base_motion_2d_custom(start_conf,
                        end_conf,
                        base_limits,
                        map_2d,
                        occupancy_range,
                        grid_resolution,
                        robot_footprint_radius_in_map,
                        weights=1 * np.ones(3),
                        resolutions=0.05 * np.ones(3),
                        min_goal_dist=-0.02,
                        algorithm='birrt',
                        optimize_iter=0,
                        visualize_planning=True,
                        visualize_result=True,
                        metric2map=None,
                        flip_vertically=False,
                        upsampling_factor = 4,
                        **kwargs):
    """
    Performs motion planning for a robot base in 2D
    :param start_conf: start config of robot
    :param end_conf: End configuration, 3D vector of the goal (x, y, theta)
    :param base_limits: Limits to sample new configurations
    :param map_2d: Map to check collisions for a sampled location
    :param occupancy_range: size of the occupancy map in metric units (assumed square), e.g., meters from left to right
           borders of the occupancy map
    :param grid_resolution: resolution/size of the occupancy grid image (pixels, expected to be squared)
           together with the occupancy_range provides the size of each pixel in metric units:
           occupancy_range/grid_resolution [m/pixel]
    :param robot_footprint_radius_in_map: Number of pixels that the robot base occupies in the occupancy map
    :param obstacles: list of obstacles to check for collisions if using pybullet as collision checker
    :param weights: Weights to compute the distance between base configurations:
            d_12 = sqrt(w1*(x1-x2)^2 + w2*(y1-y2)^2 + w3*(theta1-theta2)^2)
    :param resolutions: step-size to extend between points
    :param max_distance: maximum distance to check collisions if using pybullet as collision checker
    :param min_goal_dist: minimum distance between the current robot location and the goal for doing planning
    :param algorithm: planning algorithm to use
    :param optimize_iter: iterations of the optimizer to run after a path has been found, to smooth it out
    :param visualize_planning: boolean. To visualize the planning process (tested with birrt only)
    :param visualize_result: To visualize the planning results (tested with birrt only)
    :param metric2map: Function to map points from metric space into pixels on the map
    :param flip_vertically: If the image needs to be flipped (for global maps)
    :param use_pb_for_collisions:  If pybullet is used to check for collisions. If not, we use the local or global 2D
           traversable map
    :param upsampling_factor: Upscaling factor to enlarge the maps
    :param kwargs:
    :return: Path (if found)
    """
    # Sampling function. Generates sample configurations within the limits
    def sample_fn():
        x, y = np.random.uniform(*base_limits)
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        return (x, y, theta)

    # Function that measures the difference between two base configuration.
    # It returns a 3D vector: diff in x, y, and theta
    difference_fn = get_base_difference_fn()
    # Function that measures the distance between two base configurations. It returns
    # d_12 = sqrt(w1*(x1-x2)^2 + w2*(y1-y2)^2 + w3*(theta1-theta2)^2)
    # diff theta is computed using circular difference
    distance_fn = get_base_distance_fn(weights=weights)

    # Function that creates candidates of points to extend a path from q1 towards q2
    def extend_fn(q1, q2):
        target_theta = np.arctan2(q2[1] - q1[1], q2[0] - q1[0]) # Amount of rotation requested

        n1 = int(np.abs(circular_difference(target_theta, q1[2]) / resolutions[2])) + 1
        n3 = int(np.abs(circular_difference(q2[2], target_theta) / resolutions[2])) + 1
        steps2 = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        n2 = int(np.max(steps2)) + 1

        # First interpolate between the initial point with initial orientation, and target orientation
        for i in range(n1+1):
            q = (i / (n1)) * np.array(difference_fn((q1[0], q1[1], target_theta), q1)) + np.array(q1)
            q = tuple(q)
            yield q

        # Then interpolate between initial point and goal point, keeping the target orientation
        for i in range(n2):
            q = (i / (n2)) * np.array(
                difference_fn((q2[0], q2[1], target_theta), (q1[0], q1[1], target_theta))) + np.array(
                (q1[0], q1[1], target_theta))
            q = tuple(q)
            yield q

        # Finally, interpolate between the final point with target orientation and final point with final orientation
        for i in range(n3+1):
            q = (i / (n3)) * np.array(difference_fn(q2, (q2[0], q2[1], target_theta))) + np.array(
                (q2[0], q2[1], target_theta))
            q = tuple(q)
            yield q

    

    # Do not plan for goals that are very close by
    # This makes it impossible to "plan" for pure rotations. Use a negative min_goal_dist to allow pure rotations
    if np.abs(start_conf[0] - end_conf[0]) < min_goal_dist and np.abs(start_conf[1] - end_conf[1]) < min_goal_dist:
        log.debug("goal is too close to the initial position. Returning")
        return None

    def transform_point_to_occupancy_map(q):
        if metric2map is None: # a local occupancy map is used
            # Vector connecting robot location and the query confs. in metric absolute space
            delta = np.array(q)[:2] - np.array(start_conf)[:2]

            # Initial orientation of the base
            theta = start_conf[2]
            # X axis of the robot given the initial orientation (x is forward)
            x_dir = np.array([np.sin(theta), -np.cos(theta)])
            # Y axis of the robot given the initial orientation (y is to the right of the robot)
            y_dir = np.array([np.cos(theta), np.sin(theta)])

            # Place the query configuration in robot's reference frame (metric robot-relative space)
            end_in_start_frame = [x_dir.dot(delta), y_dir.dot(delta)]

            # Find the corresponding pixel for the query in the occupancy map, assuming the robot is at the center
            # (image-space)
            pts = np.array(end_in_start_frame) * (grid_resolution/occupancy_range) + grid_resolution / 2
            pts = pts.astype(np.int32)
        else:
            pts = metric2map(np.array(q[0:2]))

        log.debug("original point {} and in image: {}".format(q, pts))
        return pts

    # Draw the initial situation of the planning problem: src, dst and occupancy map
    if visualize_planning or visualize_result:
        planning_map_2d = cv2.cvtColor(map_2d, cv2.COLOR_GRAY2RGB)
        origin_in_image = transform_point_to_occupancy_map(start_conf)
        goal_in_image = transform_point_to_occupancy_map(end_conf)
        cv2.circle(planning_map_2d, [origin_in_image[1], origin_in_image[0]], radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(planning_map_2d, [goal_in_image[1], goal_in_image[0]], radius=3, color=(255, 0, 255), thickness=-1)
        if visualize_planning:
            cv2.namedWindow("Planning Problem")
            planning_map_2d_upsampled = cv2.resize(planning_map_2d, None, None, upsampling_factor, upsampling_factor, cv2.INTER_NEAREST)
            # We need to flip vertically if we use the global map. Points are already in the right frame
            if flip_vertically:
                planning_map_2d_upsampled = cv2.flip(planning_map_2d_upsampled, 0)
            cv2.imshow("Planning Problem", planning_map_2d_upsampled)
            cv2.waitKey(1)

    # Auxiliary function to draw two points and the line connecting them, to draw paths
    def draw_path(q1, q2, color, image=None):
        pt1 = transform_point_to_occupancy_map(q1)
        pt2 = transform_point_to_occupancy_map(q2)
        if image is None:
            cv2.circle(planning_map_2d, [pt2[1], pt2[0]], radius=1, color=color, thickness=-1)
            cv2.line(planning_map_2d, [pt1[1], pt1[0]], [pt2[1], pt2[0]], color, thickness=1, lineType=8)
            planning_map_2d_upsampled = cv2.resize(planning_map_2d, None, None, upsampling_factor, upsampling_factor, cv2.INTER_NEAREST)
            # We need to flip vertically if we use the global map. Points are already in the right frame
            if flip_vertically:
                planning_map_2d_upsampled = cv2.flip(planning_map_2d_upsampled, 0)
            cv2.imshow("Planning Problem", planning_map_2d_upsampled)
            cv2.waitKey(1)
        else:
            cv2.line(image, [pt1[1], pt1[0]], [pt2[1], pt2[0]], color, thickness=1, lineType=8)

    # Auxiliary function to draw a point
    def draw_point(q1, color, not_in_image=False, radius=1):
        if not_in_image:
            q = transform_point_to_occupancy_map(q1)
        else:
            q = q1
        cv2.circle(planning_map_2d, (q[1], q[0]), radius=radius, color=color, thickness=-1)
        planning_map_2d_upsampled = cv2.resize(planning_map_2d, None, None, upsampling_factor, upsampling_factor, cv2.INTER_NEAREST)
        # We need to flip vertically if we use the global map. Points are already in the right frame
        if flip_vertically:
            planning_map_2d_upsampled = cv2.flip(planning_map_2d_upsampled, 0)
        cv2.imshow("Planning Problem", planning_map_2d_upsampled)
        cv2.waitKey(1)

    # Function to check collisions for a given configuration q
    def collision_fn(q):
        # TODO: update this function
        pts = transform_point_to_occupancy_map(q)

        if visualize_planning:
            draw_point(pts, (0, 155, 155))
            cv2.waitKey(1)

        # Use local or global map
        # If the point is outside of the map/occupancy map, then we return "collision"
        if pts[0] < robot_footprint_radius_in_map or pts[1] < robot_footprint_radius_in_map \
            or pts[0] > grid_resolution - robot_footprint_radius_in_map - 1 or pts[
                1] > grid_resolution - robot_footprint_radius_in_map - 1:
            return True

        # We create a mask using the robot size (assumed base is circular) to check collisions around a point
        mask = np.zeros((robot_footprint_radius_in_map * 2 + 1,
                            robot_footprint_radius_in_map * 2 + 1))
        cv2.circle(mask, (robot_footprint_radius_in_map, robot_footprint_radius_in_map), robot_footprint_radius_in_map,
                    1, -1)
        mask = mask.astype(np.bool)

        # Check if all the points where the shifted mask of the robot base overlaps with the occupancy map are
        # marked as FREESPACE, and return false if not
        map_2d_around_robot = map_2d[pts[0] - robot_footprint_radius_in_map : pts[0] + robot_footprint_radius_in_map + 1,
                        pts[1] - robot_footprint_radius_in_map : pts[1] + robot_footprint_radius_in_map + 1]

        in_collision = not np.all(map_2d_around_robot[mask] == OccupancyGridState.FREESPACE)

        if visualize_planning:
            planning_map_2d_cpy = planning_map_2d[
                                  pts[0] - robot_footprint_radius_in_map: pts[0] + robot_footprint_radius_in_map + 1,
                                  pts[1] - robot_footprint_radius_in_map: pts[1] + robot_footprint_radius_in_map + 1].copy()
            draw_point(pts, 1, radius=robot_footprint_radius_in_map)
            cv2.waitKey(10) #Extra wait to visualize better the process

        log.debug("In collision? {}".format(in_collision))

        if visualize_planning:
            planning_map_2d[pts[0] - robot_footprint_radius_in_map: pts[0] + robot_footprint_radius_in_map + 1, pts[1] - robot_footprint_radius_in_map: pts[1] + robot_footprint_radius_in_map + 1] = planning_map_2d_cpy
            planning_map_2d_upsampled = cv2.resize(planning_map_2d, None, None, upsampling_factor, upsampling_factor, cv2.INTER_NEAREST)
            # We need to flip vertically if we use the global map. Points are already in the right frame
            if flip_vertically:
                planning_map_2d_upsampled = cv2.flip(planning_map_2d_upsampled, 0)
            cv2.imshow("Planning Problem", planning_map_2d_upsampled)
            cv2.waitKey(10)

        return in_collision

    # Do not plan if the initial pose is in collision
    if collision_fn(start_conf):
        log.debug("Warning: initial configuration is in collision")
        return None

    # Do not plan if the final pose is in collision
    if collision_fn(end_conf):
        log.debug("Warning: end configuration is in collision")
        return None

    if algorithm == 'direct':
        path = direct_path(start_conf, end_conf, extend_fn, collision_fn)
    elif algorithm == 'birrt':
        path = birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, draw_path=[None, draw_path][visualize_planning], draw_point=[None, draw_point][visualize_planning], **kwargs)
    elif algorithm == 'rrt_star':
        path = rrt_star(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=5000, **kwargs)
    elif algorithm == 'rrt':
        path = rrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, iterations=5000, **kwargs)
    elif algorithm == 'lazy_prm':
        path = lazy_prm_replan_loop(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, [250, 500, 1000, 2000, 4000, 4000], **kwargs)
    else:
        path = None

    if optimize_iter > 0 and path is not None:
        log.info("Optimizing the path found")
        path = optimize_path(path, extend_fn, collision_fn, iterations=optimize_iter)

    if visualize_result and path is not None:
        cv2.namedWindow("Resulting Plan")
        result_map_2d = cv2.cvtColor(map_2d, cv2.COLOR_GRAY2RGB)
        origin_in_image = transform_point_to_occupancy_map(start_conf)
        goal_in_image = transform_point_to_occupancy_map(end_conf)
        cv2.circle(result_map_2d, [origin_in_image[1], origin_in_image[0]], radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(result_map_2d, [goal_in_image[1], goal_in_image[0]], radius=3, color=(255, 0, 255), thickness=-1)
        for idx in range(len(path)-1):
            draw_path(path[idx], path[idx+1], (0,0,155), image=result_map_2d)
        result_map_2d_scaledup = cv2.resize(result_map_2d, None, None, upsampling_factor, upsampling_factor, cv2.INTER_NEAREST)
        # We need to flip vertically if we use the global map. Points are already in the right frame
        if flip_vertically:
            result_map_2d_scaledup = cv2.flip(result_map_2d_scaledup, 0)
        cv2.imshow("Resulting Plan", result_map_2d_scaledup)
        cv2.waitKey(10)

    return path