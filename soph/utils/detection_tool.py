import numpy as np
from soph.utils.pcd_dict import PointCloudDict
class DetectionTool:
    def __init__(self) -> None:
        self.pois = []
        self.definitive_detections = []
        self.theta_threshold = np.pi / 6
        self.dist_threshold = 1.0


    def register_new_poi(self, new_poi):
        if not self.unique_poi(new_poi): return False
        self.pois.append(new_poi)
        return True

    def remove_close_pois(self, state):
        self.pois = list(filter(lambda x: not self.is_similar(state, x), self.pois))

    def register_definitive_detection(self, points):
        if self.already_detected(points): return None
        new_detection = DefinitiveDetection(points)
        self.definitive_detections.append(new_detection)
        self.pois = list(filter(lambda x: not new_detection.would_detect(x), self.pois))
        return new_detection

    def is_similar(self, poi, new_poi):
        # detection has format [x, y, theta]
        if np.abs(new_poi[2] - poi[2]) > self.theta_threshold: return False
        if np.linalg.norm(new_poi[:2] - poi[:2]) > self.dist_threshold: return False
        return True

    def unique_poi(self, new_poi):
        for def_detection in self.definitive_detections:
            if def_detection.would_detect(new_poi): return False
        for poi in self.pois:
            if self.is_similar(poi, new_poi): return False
        return True

    def already_detected(self, points):
        center = np.average(np.vstack(points), axis=0)
        for def_detection in self.definitive_detections:
            if def_detection.equivalent_point(center[:2]):
                def_detection.extend(points)
                return True
        return False

    def closest_poi(self, position):
        dist2 = np.inf
        closest = None
        for poi in self.pois:
            d2 = (position[0] - poi[0])**2 + (position[1] - poi[1])**2
            if d2 < dist2:
                dist2 = d2
                closest = poi
        return closest

class DefinitiveDetection:
    def __init__(self, points, similarity_threshold = 0.25):
        self.low_res_point_cloud = PointCloudDict(1, 2)
        self.extend(points)
        self.similarity_threshold = similarity_threshold

    def extend(self, points):
        for point in points:
            self.low_res_point_cloud.insert(point)
        center = np.average(self.low_res_point_cloud.voxel_point_array(), axis=0)
        self.position = np.array(center[:2])

    def contains(self, points):
        for point in points:
            if self.low_res_point_cloud.contains(point):
                self.extend(points)
                return True
        return False

    def equivalent_detection(self, detection):
        return np.linalg.norm(self.position - detection.position) < self.similarity_threshold

    def equivalent_point(self, point):
        return np.linalg.norm(self.position - point) < self.similarity_threshold

    def would_detect(self, poi):
        unitv = np.array([np.cos(poi[2]), np.sin(poi[2])])
        px = self.position[0]
        py = self.position[1]
        x0 = poi[0]
        y0 = poi[1]
        u0 = unitv[0]
        v0 = unitv[1]

        a = ((px + v0 * py / u0) - (x0 + v0 * y0 / u0))/(u0 + v0 * v0 / u0)
        if a < 0: return False
        dist = (x0 + a*u0 - px) / v0
        return dist < self.similarity_threshold

