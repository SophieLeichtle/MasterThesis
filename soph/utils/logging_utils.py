import logging
import os
import cv2
import time
from soph import logs_path, DEFAULT_FOOTPRINT_RADIUS


def initiate_logging(log_name):
    datestr = time.strftime("%Y-%m-%d")
    timestr = time.strftime("%H-%M-%S")
    log_dir = os.path.join(logs_path, datestr, timestr)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, log_name)
    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(filename), logging.StreamHandler()],
    )
    return log_dir


def save_map(
    log_dir, robot_state, map, detection_tool, current_plan=[], frontier_line=[]
):
    filename = time.strftime("%H-%M-%S-map.png")
    map_cv = cv2.cvtColor(map.grid * 255, cv2.COLOR_GRAY2RGB)
    robot_pos_in_map = map.m_to_px(robot_state[:2])
    base_radius_in_map = int(DEFAULT_FOOTPRINT_RADIUS * map.m_to_pix_ratio)
    cv2.circle(
        map_cv,
        [int(robot_pos_in_map[1]), int(robot_pos_in_map[0])],
        radius=base_radius_in_map,
        color=(255, 0, 0),
        thickness=-1,
    )
    for def_det in detection_tool.definitive_detections:
        det_pos_in_map = map.m_to_px(def_det.position)
        cv2.circle(
            map_cv,
            [int(det_pos_in_map[1]), int(det_pos_in_map[0])],
            radius=3,
            color=(0, 255, 0),
            thickness=-1,
        )
    for poi in detection_tool.pois:
        poi_in_map = map.m_to_px(poi[:2])
        cv2.circle(
            map_cv,
            [int(poi_in_map[1]), int(poi_in_map[0])],
            radius=3,
            color=(0, 0, 255),
            thickness=-1,
        )
    for point in current_plan:
        point_in_map = map.m_to_px(point[:2])
        cv2.circle(
            map_cv,
            [int(point_in_map[1]), int(point_in_map[0])],
            radius=1,
            color=(255, 0, 255),
            thickness=-1,
        )
    for point in frontier_line:
        cv2.circle(
            map_cv,
            [int(point[1]), int(point[0])],
            radius=1,
            color=(0, 255, 255),
            thickness=-1,
        )
    cv2.imwrite(os.path.join(log_dir, filename), map_cv)
    logging.info("Saved Occupancy Map as " + filename)


def save_nav_map(log_dir, map, nav_graph):
    filename = time.strftime("%H-%M-%S-nav.png")
    map_cv = cv2.cvtColor(map.grid * 255, cv2.COLOR_GRAY2RGB)
    for node in nav_graph.nodes:
        if node.parent is not None:
            pos_in_map = map.m_to_px(node.position)
            parent_in_map = map.m_to_px(node.parent.position)
            cv2.line(
                map_cv,
                [int(pos_in_map[1]), int(pos_in_map[0])],
                [int(parent_in_map[1]), int(parent_in_map[0])],
                color=(0, 255, 0),
                thickness=1,
            )
            cv2.circle(
                map_cv,
                [int(pos_in_map[1]), int(pos_in_map[0])],
                radius=2,
                color=(0, 255, 0),
                thickness=-1,
            )
    root_in_map = map.m_to_px(nav_graph.root.position)
    cv2.circle(
        map_cv,
        [int(root_in_map[1]), int(root_in_map[0])],
        radius=3,
        color=(255, 0, 0),
        thickness=-1,
    )
    cv2.imwrite(os.path.join(log_dir, filename), map_cv)


def save_map_combo(
    log_dir,
    robot_state,
    map,
    detection_tool,
    nav_graph,
    current_plan=[],
    frontier_line=[],
):
    filename = time.strftime("%H-%M-%S-combo.png")
    map_cv = cv2.cvtColor(map.grid * 255, cv2.COLOR_GRAY2RGB)
    robot_pos_in_map = map.m_to_px(robot_state[:2])
    base_radius_in_map = int(DEFAULT_FOOTPRINT_RADIUS * map.m_to_pix_ratio)
    # Robot Base: Blue
    cv2.circle(
        map_cv,
        [int(robot_pos_in_map[1]), int(robot_pos_in_map[0])],
        radius=base_radius_in_map,
        color=(255, 0, 0),
        thickness=-1,
    )
    # Definitive Detections: Orange
    for def_det in detection_tool.definitive_detections:
        det_pos_in_map = map.m_to_px(def_det.position)
        cv2.circle(
            map_cv,
            [int(det_pos_in_map[1]), int(det_pos_in_map[0])],
            radius=3,
            color=(0, 127, 255),
            thickness=-1,
        )
    # POIs: Red
    for poi in detection_tool.pois:
        poi_in_map = map.m_to_px(poi[:2])
        cv2.circle(
            map_cv,
            [int(poi_in_map[1]), int(poi_in_map[0])],
            radius=3,
            color=(0, 0, 255),
            thickness=-1,
        )
    # Current Plan: Pink
    for point in current_plan:
        point_in_map = map.m_to_px(point[:2])
        cv2.circle(
            map_cv,
            [int(point_in_map[1]), int(point_in_map[0])],
            radius=1,
            color=(255, 0, 255),
            thickness=-1,
        )
    # Frontier Line: Yellow
    for point in frontier_line:
        cv2.circle(
            map_cv,
            [int(point[1]), int(point[0])],
            radius=1,
            color=(0, 255, 255),
            thickness=-1,
        )
    # Node Graph: Green
    for node in nav_graph.nodes:
        if node.parent is not None:
            pos_in_map = map.m_to_px(node.position)
            parent_in_map = map.m_to_px(node.parent.position)
            cv2.line(
                map_cv,
                [int(pos_in_map[1]), int(pos_in_map[0])],
                [int(parent_in_map[1]), int(parent_in_map[0])],
                color=(0, 255, 0),
                thickness=1,
            )
            cv2.circle(
                map_cv,
                [int(pos_in_map[1]), int(pos_in_map[0])],
                radius=2,
                color=(0, 255, 0),
                thickness=-1,
            )
    root_in_map = map.m_to_px(nav_graph.root.position)
    cv2.circle(
        map_cv,
        [int(root_in_map[1]), int(root_in_map[0])],
        radius=3,
        color=(0, 127, 0),
        thickness=-1,
    )
    cv2.imwrite(os.path.join(log_dir, filename), map_cv)


def save_map_rt_rrt_star(
    file_name,
    robot_pos,
    occupancy_map,
    rt_rrt_star,
    detection_tool=None,
    current_frontier=[],
):
    map_cv = cv2.cvtColor(occupancy_map.grid * 255, cv2.COLOR_GRAY2RGB)
    for cluster in rt_rrt_star.node_clusters.clusters.values():
        for node in cluster:
            if node.parent() is not None:
                pos_in_map = occupancy_map.m_to_px(node.position())
                parent_in_map = occupancy_map.m_to_px(node.parent().position())
                cv2.line(
                    map_cv,
                    [int(pos_in_map[1]), int(pos_in_map[0])],
                    [int(parent_in_map[1]), int(parent_in_map[0])],
                    color=(0, 255, 0),
                    thickness=1,
                )
    root_in_map = occupancy_map.m_to_px(rt_rrt_star.root.position())
    cv2.circle(
        map_cv,
        [int(root_in_map[1]), int(root_in_map[0])],
        radius=3,
        color=(255, 0, 255),
        thickness=-1,
    )

    if detection_tool is not None:
        # Definitive Detections: Orange
        for def_det in detection_tool.definitive_detections:
            det_pos_in_map = occupancy_map.m_to_px(def_det.position)
            cv2.circle(
                map_cv,
                [int(det_pos_in_map[1]), int(det_pos_in_map[0])],
                radius=3,
                color=(0, 127, 255),
                thickness=-1,
            )
        # POIs: Red
        for poi in detection_tool.pois:
            poi_in_map = occupancy_map.m_to_px(poi[:2])
            cv2.circle(
                map_cv,
                [int(poi_in_map[1]), int(poi_in_map[0])],
                radius=3,
                color=(0, 0, 255),
                thickness=-1,
            )

    if rt_rrt_star.dummy_goal_node is not None:
        goal_in_map = occupancy_map.m_to_px(rt_rrt_star.dummy_goal_node.position())
        cv2.circle(
            map_cv,
            [int(goal_in_map[1]), int(goal_in_map[0])],
            radius=3,
            color=(0, 255, 255),
            thickness=-1,
        )
    for point in current_frontier:
        cv2.circle(
            map_cv,
            [int(point[1]), int(point[0])],
            radius=1,
            color=(0, 255, 255),
            thickness=-1,
        )

    robot_pos_in_map = occupancy_map.m_to_px(robot_pos)
    cv2.circle(
        map_cv,
        [int(robot_pos_in_map[1]), int(robot_pos_in_map[0])],
        radius=2,
        color=(255, 0, 0),
        thickness=-1,
    )
    cv2.imwrite(file_name, map_cv)


def save_map_rt_rrt_star_detailed(
    file_name,
    occupancy_map,
    rt_rrt_star,
):

    map_cv = cv2.cvtColor(occupancy_map.grid * 255, cv2.COLOR_GRAY2RGB)
    for cluster in rt_rrt_star.node_clusters.clusters.values():
        for node in cluster:
            if node.parent() is not None:
                pos_in_map = occupancy_map.m_to_px(node.position())
                parent_in_map = occupancy_map.m_to_px(node.parent().position())
                cv2.line(
                    map_cv,
                    [int(pos_in_map[1]), int(pos_in_map[0])],
                    [int(parent_in_map[1]), int(parent_in_map[0])],
                    color=(0, 255, 0),
                    thickness=1,
                )
    root_in_map = occupancy_map.m_to_px(rt_rrt_star.root.position())

    for node in rt_rrt_star.Q_s:
        pos_in_map = occupancy_map.m_to_px(node.position())
        cv2.circle(
            map_cv,
            [int(pos_in_map[1]), int(pos_in_map[0])],
            radius=2,
            color=(255, 0, 0),
        )
    for node in rt_rrt_star.visited:
        pos_in_map = occupancy_map.m_to_px(node.position())
        cv2.circle(
            map_cv,
            [int(pos_in_map[1]), int(pos_in_map[0])],
            radius=1,
            color=(255, 0, 0),
        )

    cv2.circle(
        map_cv,
        [int(root_in_map[1]), int(root_in_map[0])],
        radius=2,
        color=(255, 0, 255),
        thickness=-1,
    )

    cv2.imwrite(file_name, map_cv)
