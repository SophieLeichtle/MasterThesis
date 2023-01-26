from re import sub
import numpy as np

class PointCloudDict:

    def __init__(self, precision=4, sub_precision=1):
        assert(sub_precision == 1 or sub_precision%2 == 0)
        self.precision = precision
        self.sub_precision = sub_precision
        self.dict = {}

    def insert(self, p):
        p_tuple = self._create_tuple(p)
        self.dict[p_tuple] = p

    def remove(self, p):
        p_tuple = self._create_tuple(p)
        self.dict.pop(p_tuple)

    def contains(self, p):
        p_tuple = self._create_tuple(p)
        return p_tuple in self.dict.keys()

    def point_array(self):
        if self.dict == {}: return []
        values = self.dict.values()
        return np.vstack(values)

    def voxel_point_array(self):
        if self.dict == {}: return []
        keys = self.dict.keys()
        return np.vstack(keys)

    def _create_tuple(self, p):
        p_tuple = (round(self.sub_precision * p[0], self.precision)/self.sub_precision, round(self.sub_precision * p[1], self.precision)/self.sub_precision, round(self.sub_precision * p[2], self.precision)/self.sub_precision)
        return p_tuple