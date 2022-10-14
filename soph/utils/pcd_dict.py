from re import sub
import numpy as np

class PointCloudDict:

    def __init__(self, precision=4, sub_precision=1):
        assert(sub_precision == 1 or sub_precision%2 == 0)
        self.precision = precision
        self.sub_precision = sub_precision
        self.dict = {}

    def insert(self, p):
        tuple = (round(self.sub_precision * p[0], self.precision)/self.sub_precision, round(self.sub_precision * p[1], self.precision)/self.sub_precision, round(self.sub_precision * p[2], self.precision)/self.sub_precision)
        self.dict[tuple] = p

    def remove(self, p):
        tuple = (round(self.sub_precision * p[0], self.precision)/self.sub_precision, round(self.sub_precision * p[1], self.precision)/self.sub_precision, round(self.sub_precision * p[2], self.precision)/self.sub_precision)
        self.dict.pop(tuple)

    def point_array(self):
        if self.dict == {}: return []
        values = self.dict.values()
        return np.vstack(values)
