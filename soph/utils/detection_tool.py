import numpy as np
from soph.utils.pcd_dict import PointCloudDict
import logging
from soph.utils.utils import bbox, px_to_3d, openglf_to_wf, center_ransac
from soph.planning.motion_planning import (
    get_poi,
)


class DetectionTool:
    """
    Tool for Managing Definitive Detections and POIs
    """

    def __init__(self) -> None:
        self.pois = []
        self.definitive_detections = []
        self.theta_threshold = np.pi / 6
        self.dist_threshold = 1.0

    def register_new_poi(self, new_poi):
        """
        Register a new Point of Interest. Returns True if POI was successfully added.
        :param new_poi: POI to be added
        """
        if not self.unique_poi(new_poi):
            return False
        self.pois.append(new_poi)
        return True

    def remove_close_pois(self, state):
        """
        Remove POIs close to the given robot state
        :param state: Current state of robot
        """
        self.pois = list(filter(lambda x: not self.is_similar(state, x), self.pois))

    def register_definitive_detection(self, points, center):
        """
        Register a new definitive detection given a set of detected points.
        Checks if points are already detected. If so, return None, else return new Detection.
        :param points: List of Points making up a new detection
        """
        if self.already_detected(points, center):
            return None
        new_detection = DefinitiveDetection(points)
        self.definitive_detections.append(new_detection)
        self.pois = list(filter(lambda x: not new_detection.would_detect(x), self.pois))
        return new_detection

    def is_similar(self, poi, new_poi):
        """
        Check if two POIs are similar based on thresholds.
        :param poi, new_poi: POIs to be compared
        """
        # detection has format [x, y, theta]
        if np.abs(new_poi[2] - poi[2]) > self.theta_threshold:
            return False
        if np.linalg.norm(new_poi[:2] - poi[:2]) > self.dist_threshold:
            return False
        return True

    def unique_poi(self, new_poi):
        """
        Check if a new POI is unique, meaning it is not similar to another POI
        and would not detect an already registered Detection
        :param new_poi: new POI to be checked
        """
        for def_detection in self.definitive_detections:
            if def_detection.would_detect(new_poi):
                return False
        for poi in self.pois:
            if self.is_similar(poi, new_poi):
                return False
        return True

    def already_detected(self, points, center):
        """
        Check if points belong to an already registered detection
        :param points: points to be checked
        """
        for def_detection in self.definitive_detections:
            if def_detection.equivalent_point(center[:2]):
                # def_detection.extend(points)
                return True
            # if def_detection.contains(points):
            # return True
        return False

    def closest_poi(self, position):
        """
        Return POI that is closest to given position
        :param position: Position
        """
        dist2 = np.inf
        closest = None
        for poi in self.pois:
            new_d2 = (position[0] - poi[0]) ** 2 + (position[1] - poi[1]) ** 2
            if new_d2 < dist2:
                dist2 = new_d2
                closest = poi
        return closest

    def matches_detection(self, detection):
        for def_detection in self.definitive_detections:
            if def_detection.would_detect(detection):
                return True
        return False

    def process_detections(self, env, detections, masks):
        state = env.get_state()
        depth = state["depth"]

        new_detection = False

        for detection, mask in zip(detections, masks):
            if self.matches_detection(detection):
                continue
            masked_depth = depth[:, :, 0] * mask
            if np.count_nonzero(masked_depth) > 50:
                rmin, rmax, cmin, cmax = bbox(masked_depth)
                points = []
                t_mat = openglf_to_wf(env.robots[0])
                for row in range(rmin, rmax + 1):
                    for col in range(cmin, cmax + 1):
                        d = masked_depth[row, col]
                        if d == 0:
                            continue
                        point = px_to_3d(row, col, d, t_mat, env.config["depth_high"])
                        if point[2] > 0.05:
                            points.append(point)
                center, inliers = center_ransac(points)
                new_detection = self.register_definitive_detection(inliers, center)
                if new_detection is not None:
                    logging.info(
                        "New Detection Located at %.2f, %.2f",
                        new_detection.position[0],
                        new_detection.position[1],
                    )
                    new_detection = True
            else:
                poi = get_poi(detection)
                new = self.register_new_poi(poi)
                if new:
                    logging.info(
                        "Object Detected: New POI added at %.2f, %.2f",
                        poi[0],
                        poi[1],
                    )

        return new_detection


class DefinitiveDetection:
    """
    Class Containing the Definitive Detection Structure as well as useful functions
    """

    def __init__(self, points, similarity_threshold=0.5):
        self.low_res_point_cloud = PointCloudDict(1, 2)
        self.extend(points)
        self.similarity_threshold = similarity_threshold

    def extend(self, points):
        """
        Extend Detection by points
        :param points: points to extend detection by
        """
        for point in points:
            self.low_res_point_cloud.insert(point)
        center = np.average(self.low_res_point_cloud.voxel_point_array(), axis=0)
        self.position = np.array(center[:2])

    def contains(self, points):
        """
        Check if Detection already contains points. Returns true if at least one Point is contained
        :param points: points to check
        """
        for point in points:
            if self.low_res_point_cloud.contains(point):
                # self.extend(points)
                return True
        return False

    def equivalent_detection(self, detection):
        """
        Check if two detections are equivalent based on threshold
        """
        return (
            np.linalg.norm(self.position - detection.position)
            < self.similarity_threshold
        )

    def equivalent_point(self, point):
        """
        Check if a point is equivalent to center of detection based on thresholds
        """
        return np.linalg.norm(self.position - point) < self.similarity_threshold

    def would_detect(self, poi):
        """
        Check if a POI would detect this detection
        """
        unitv = np.array([np.cos(poi[2]), np.sin(poi[2])])
        p_x = self.position[0]
        p_y = self.position[1]
        x_0 = poi[0]
        y_0 = poi[1]
        u_0 = unitv[0]
        v_0 = unitv[1]

        a = ((p_x + v_0 * p_y / u_0) - (x_0 + v_0 * y_0 / u_0)) / (
            u_0 + v_0 * v_0 / u_0
        )
        if a < 0:
            return False
        dist = (x_0 + a * u_0 - p_x) / v_0
        return dist < 0.25
