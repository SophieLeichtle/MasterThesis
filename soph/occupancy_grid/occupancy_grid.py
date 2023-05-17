import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from soph.utils.utils import pixel_to_point, bbox
from soph.utils.bresenham import plotLine
from scipy.ndimage import binary_erosion, binary_dilation

from igibson.utils.constants import OccupancyGridState
from soph import DEFAULT_FOOTPRINT_RADIUS


class OccupancyGrid2D:
    def __init__(self, half_size=250, m_to_pix=128.0 / 5.0):
        self.half_size = half_size
        self.grid = np.zeros((self.half_size * 2 + 1, self.half_size * 2 + 1)).astype(
            np.float32
        )
        self.grid.fill(OccupancyGridState.UNKNOWN)
        self.origin = np.array([half_size, half_size])
        self.m_to_pix_ratio = m_to_pix
        self.coordinate_transform = np.array(((0, -1), (1, 0)))

    def px_to_m(self, point):
        """
        Convert a pixel to meters. Pixel does not need to be an integer.
        """
        return np.dot(
            np.linalg.inv(self.coordinate_transform),
            (point - self.origin) / self.m_to_pix_ratio,
        )

    def m_to_px(self, position):
        """
        Convert a position in meters to pixels. Returns floating point pixel.
        """
        return (
            np.dot(self.coordinate_transform, position) * self.m_to_pix_ratio
            + self.origin
        )

    def update_with_grid(self, occupancy_grid, position, theta):
        """
        Update the occupancy grid using an external occupancy grid.
        Assumes the m to px ratio is the same for both.

        :param occupancy_grid: external occupancy grid
        :param position: robot position in world space
        :param theta: robot theta in world space
        """
        robot_pos_in_map = self.m_to_px(position)

        # Compute Rotation Matrix
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        # Input Grid must be square
        assert occupancy_grid.shape[0] == occupancy_grid.shape[1]
        grid_size = occupancy_grid.shape[0]

        # Update Grid
        for r in range(0, grid_size):
            for c in range(0, grid_size):
                if occupancy_grid[r][c] == OccupancyGridState.UNKNOWN:
                    continue
                point = robot_pos_in_map + np.dot(
                    R, [r - grid_size // 2, c - grid_size // 2]
                )
                if occupancy_grid[r][c] == OccupancyGridState.OBSTACLES:
                    rounded_point = [int(round(point[0])), int(round(point[1]))]
                    self.grid[rounded_point[0], rounded_point[1]] = occupancy_grid[r][c]
                    continue
                for x in np.floor(point[1]).astype(np.int32), np.ceil(point[1]).astype(
                    np.int32
                ):
                    for y in np.floor(point[0]).astype(np.int32), np.ceil(
                        point[0]
                    ).astype(np.int32):
                        if self.grid[y][x] != OccupancyGridState.OBSTACLES:
                            self.grid[y][x] = occupancy_grid[r][c]

    def update_with_grid_direct(self, occupancy_grid, position):
        robot_pos_in_map = self.m_to_px(position)

        # Input Grid must be square
        assert occupancy_grid.shape[0] == occupancy_grid.shape[1]
        grid_size = occupancy_grid.shape[0]

        for r in range(0, grid_size):
            for c in range(0, grid_size):
                if occupancy_grid[r][c] == OccupancyGridState.UNKNOWN:
                    continue
                point = robot_pos_in_map + np.array(
                    [r - grid_size // 2, c - grid_size // 2]
                )

                if occupancy_grid[r][c] == OccupancyGridState.OBSTACLES:
                    rounded_point = [int(round(point[0])), int(round(point[1]))]
                    self.grid[
                        rounded_point[0], rounded_point[1]
                    ] = OccupancyGridState.OBSTACLES
                    continue

                for x in np.floor(point[1]).astype(np.int32), np.ceil(point[1]).astype(
                    np.int32
                ):
                    for y in np.floor(point[0]).astype(np.int32), np.ceil(
                        point[0]
                    ).astype(np.int32):
                        if self.grid[y][x] != OccupancyGridState.OBSTACLES:
                            self.grid[y][x] = OccupancyGridState.FREESPACE

    def update_with_points(self, points):
        """
        Update the occupancy grid using 3d points

        :param points: list of points in world space
        """
        for point in points:
            point_in_map = self.m_to_px(point[:2])
            rounded_point = [int(round(point_in_map[0])), int(round(point_in_map[1]))]
            if (
                self.grid[rounded_point[0], rounded_point[1]]
                != OccupancyGridState.UNKNOWN
            ):
                self.grid[
                    rounded_point[0], rounded_point[1]
                ] = OccupancyGridState.OBSTACLES

    def update_from_depth(self, env, depth, samplesize=1000):
        """
        Update with a depth image using sampling.
        If samplesize == -1, the whole image is used
        """
        points = []
        if samplesize == -1:
            for r in range(depth.shape[0]):
                for c in range(depth.shape[1]):
                    d = depth[r, c, 0]
                    if d == 0:
                        continue
                    p = pixel_to_point(env, r, c, d)
                    if p[2] > 0.2:
                        points.append(p)
        else:
            rows = np.random.randint(depth.shape[0], size=samplesize)
            columns = np.random.randint(depth.shape[1], size=samplesize)
            for it in range(0, samplesize):
                d = depth[rows[it], columns[it], 0]
                if d == 0:
                    continue
                p = pixel_to_point(env, rows[it], columns[it], d)
                if p[2] > 0.2:
                    points.append(p)
        self.update_with_points(points)

    def check_if_free(self, position, base_radius=DEFAULT_FOOTPRINT_RADIUS):
        """
        Check if a certain position is free given a robot base radius

        :param position: position of base in world space
        :param base_radius: robot base radius
        """
        # If center point is not free can skip more intensive calculations
        robot_pos_in_map = self.m_to_px(position)
        if (
            self.grid[int(robot_pos_in_map[0]), int(robot_pos_in_map[1])]
            != OccupancyGridState.FREESPACE
        ):
            return False

        filter = np.zeros_like(self.grid)
        # Careful! Opencv uses different indexing so transformation is different
        robot_pos_in_map_cv = (
            np.array([position[0], -position[1]]) * self.m_to_pix_ratio + self.origin
        )

        base_radius_in_map = int(1.1 * base_radius * self.m_to_pix_ratio)
        cv2.circle(
            img=filter,
            center=robot_pos_in_map_cv.astype(np.int32),
            radius=base_radius_in_map,
            color=1,
            thickness=-1,
        )

        points = self.grid[filter == 1]
        return (
            OccupancyGridState.OBSTACLES not in points
            and OccupancyGridState.UNKNOWN not in points
        )

    def check_new_information(
        self, position, theta, lin_range, ang_range, visualize=False
    ):
        """
        Estimates information gain for a given robot configuration by casting rays.

        :param position, theta: robot position and theta
        :param lin_range: linear range of the lidar
        :param ang_range: angular range of the lidar
        :param visualize: if true, visualize information gain
        """
        robot_pos_in_map = self.m_to_px(position)

        angles = np.linspace(theta - ang_range / 2, theta + ang_range / 2, 30)
        ranges = np.linspace(0, lin_range * self.m_to_pix_ratio, 20)

        new_info = 0

        grid_copy = self.grid.copy()

        for angle in angles:
            unit_v = np.array([-np.sin(angle), np.cos(angle)])
            for range in ranges:
                point = (robot_pos_in_map + range * unit_v).astype(np.int32)
                value = self.grid[point[0], point[1]]
                grid_copy[point[0], point[1]] = 2
                if value == OccupancyGridState.OBSTACLES:
                    break
                if value == OccupancyGridState.UNKNOWN:
                    new_info += 1

        if visualize:
            plt.figure()
            plt.imshow(grid_copy)
            plt.show()
        return new_info

    def new_information_simple(self, position, lidar_range):
        pos_in_map = self.m_to_px(position)
        range_in_map = self.m_to_pix_ratio * lidar_range

        filter = np.zeros_like(self.grid)
        cv2.circle(
            img=filter,
            center=[int(pos_in_map[1]), int(pos_in_map[0])],
            radius=int(range_in_map),
            color=1,
            thickness=-1,
        )

        points = self.grid[filter == 1]
        return np.count_nonzero(points == OccupancyGridState.UNKNOWN)

    def line_of_sight(self, pos_in_map, goal_in_map):

        line = plotLine(
            int(pos_in_map[0]),
            int(pos_in_map[1]),
            int(goal_in_map[0]),
            int(goal_in_map[1]),
        )
        for point in line:
            if self.grid[point[0], point[1]] == OccupancyGridState.OBSTACLES:
                return False
        return True

    def line_of_sight_complex(self, p1, p2, base_radius=DEFAULT_FOOTPRINT_RADIUS):
        p1_in_map_cv = (
            np.array([p1[0], -p1[1]]) * self.m_to_pix_ratio + self.origin
        ).astype(np.int32)
        p2_in_map_cv = (
            np.array([p2[0], -p2[1]]) * self.m_to_pix_ratio + self.origin
        ).astype(np.int32)

        t = p2_in_map_cv - p1_in_map_cv
        if np.linalg.norm(t) == 0:
            return self.check_if_free(p1, base_radius)

        base_radius_in_map = int(1.1 * base_radius * self.m_to_pix_ratio)
        n = (np.array([t[1], -t[0]]) / np.linalg.norm(t) * (base_radius_in_map)).astype(
            np.int32
        )

        poly = []
        poly.append(p1_in_map_cv + n)
        poly.append(poly[-1] + t)
        poly.append(poly[-1] - 2 * n)
        poly.append(poly[-1] - t)
        poly = np.array(poly).reshape(1, -1, 2)

        filter = np.zeros_like(self.grid)
        cv2.circle(
            img=filter,
            center=p1_in_map_cv,
            radius=base_radius_in_map,
            color=1,
            thickness=-1,
        )
        cv2.circle(
            img=filter,
            center=p2_in_map_cv,
            radius=base_radius_in_map,
            color=1,
            thickness=-1,
        )
        cv2.fillPoly(img=filter, pts=poly, color=1, lineType=1)

        points = self.grid[filter == 1]
        return (
            OccupancyGridState.OBSTACLES not in points
            and OccupancyGridState.UNKNOWN not in points
        )

    def sample_uniform(self):
        rmin, rmax, cmin, cmax = bbox(self.grid != OccupancyGridState.UNKNOWN)
        p_r = random.uniform(0, 1)
        r = int(np.round(rmin + (rmax - rmin) * p_r))
        p_c = random.uniform(0, 1)
        c = int(np.round(cmin + (cmax - cmin) * p_c))
        while self.grid[r, c] != OccupancyGridState.FREESPACE:
            p_r = random.uniform(0, 1)
            r = int(np.round(rmin + (rmax - rmin) * p_r))
            p_c = random.uniform(0, 1)
            c = int(np.round(cmin + (cmax - cmin) * p_c))
        return self.px_to_m(np.array([r, c]))

    def free_space(self):
        free_pixels = np.count_nonzero(self.grid == OccupancyGridState.FREESPACE)
        return free_pixels * (1 / self.m_to_pix_ratio) * (1 / self.m_to_pix_ratio)

    def explored_space(self):
        explored_pixels = np.count_nonzero(self.grid != OccupancyGridState.UNKNOWN)
        return explored_pixels * (1 / self.m_to_pix_ratio) * (1 / self.m_to_pix_ratio)
