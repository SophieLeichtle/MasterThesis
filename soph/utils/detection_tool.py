import numpy as np

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
        self.pois = filter(lambda x: not self.is_similar(state, x), self.pois)

    def register_definitive_detection(self, new_detection):
        if self.already_detected(new_detection): return False
        self.definitive_detections.append(new_detection)
        self.pois = filter(lambda x: not new_detection.would_detect(x), self.pois)
        return True

    def is_similar(self, poi, new_poi):
        # detection has format [x, y, theta]
        if np.abs(new_poi[2] - poi[2]) > self.theta_threshold: return False
        if np.linalg.norm(new_poi[:2] - poi[:2]) > self.dist_threshold: return False
        return True

    def unique_poi(self, new_poi):
        for def_detection in self.definitive_detections:
            if def_detection.would_detect(poi): return True
        for poi in self.pois:
            if self.is_similar(poi, new_poi): return False
        return True

    def already_detected(self, new_detection):
        for def_detection in self.definitive_detections:
            if def_detection.equivalent(new_detection): return True
        return False

class DefinitiveDetection:
    def __init__(self, position, similarity_threshold = 0.25):
        self.position = position
        self.similarity_threshold = similarity_threshold

    def equivalent(self, detection):
        return np.linalg.norm(self.position - detection.position) < self.similarity_threshold

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

