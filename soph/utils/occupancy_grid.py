import numpy as np
import cv2
import matplotlib.pyplot as plt

class OccupancyGrid2D:

    def __init__(self, half_size=250, m_to_pix= 128.0/5.0):
        self.half_size = half_size
        self.grid = np.zeros((self.half_size*2 + 1, self.half_size*2 + 1)).astype(np.float32)
        self.grid.fill(0.5)
        self.origin = np.array([half_size, half_size])
        self.m_to_pix_ratio = m_to_pix
        self.coordinate_transform = np.array(((0, -1),(1, 0)))

    def px_to_m(self, point):
        """
        Convert a pixel to meters. Pixel does not need to be an integer.
        """
        return np.dot(np.linalg.inv(self.coordinate_transform),(point - self.origin) / self.m_to_pix_ratio)
    def m_to_px(self, position):
        """
        Convert a position in meters to pixels. Returns floating point pixel.
        """
        return np.dot(self.coordinate_transform, position) * self.m_to_pix_ratio + self.origin

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
        R = np.array(((c,-s), (s, c)))

        # Input Grid must be square
        assert(occupancy_grid.shape[0] == occupancy_grid.shape[1])
        grid_size = occupancy_grid.shape[0]

        # Update Grid
        for r in range(0,grid_size):
            for c in range(0,grid_size):
                if occupancy_grid[r][c] == 0.5:
                    continue
                point = robot_pos_in_map + np.dot(R, [r-grid_size//2,c-grid_size//2])
                for x in np.floor(point[1]).astype(np.int32), np.ceil(point[1]).astype(np.int32):
                    for y in np.floor(point[0]).astype(np.int32), np.ceil(point[0]).astype(np.int32):
                        if self.grid[y][x] != 0:
                            self.grid[y][x] = occupancy_grid[r][c]

    def update_with_points(self, points):
        """
        Update the occupancy grid using 3d points

        :param points: list of points in world space
        """
        for point in points:
            point_in_map = np.dot(self.coordinate_transform, point[:2]) * self.m_to_pix_ratio + self.origin
            for x in np.floor(point_in_map[1]).astype(np.int32), np.ceil(point_in_map[1]).astype(np.int32):
                    for y in np.floor(point_in_map[0]).astype(np.int32), np.ceil(point_in_map[0]).astype(np.int32):
                        if self.grid[y][x] != 0.5:
                            self.grid[y][x] = 0

    def check_if_free(self, position, base_radius):
        """
        Check if a certain position is free given a robot base radius

        :param position: position of base in world space
        :param base_radius: robot base radius
        """
        filter = np.zeros_like(self.grid)
        #Careful! Opencv uses different indexing so transformation is different
        robot_pos_in_map = np.array([position[0], -position[1]]) * self.m_to_pix_ratio + self.origin

        base_radius_in_map = int(base_radius * self.m_to_pix_ratio)
        cv2.circle(img=filter,center=robot_pos_in_map.astype(np.int32), radius=base_radius_in_map, color=1, thickness=-1)
        
        points = self.grid[filter==1]
        return 0 not in points and 0.5 not in points

    def check_new_information(self, position, theta, lin_range, ang_range, visualize=False):
        """
        Estimates information gain for a given robot configuration by casting rays.

        :param position, theta: robot position and theta
        :param lin_range: linear range of the lidar 
        :param ang_range: angular range of the lidar
        :param visualize: if true, visualize information gain
        """
        robot_pos_in_map = self.m_to_px(position)

        angles = np.linspace(theta - ang_range/2, theta + ang_range/2, 30)
        ranges = np.linspace(0, lin_range * self.m_to_pix_ratio, 20)

        new_info = 0

        grid_copy = self.grid.copy()

        for angle in angles:
            unit_v = np.array([-np.sin(angle), np.cos(angle)])
            for range in ranges:
                point = (robot_pos_in_map + range * unit_v).astype(np.int32)
                value = self.grid[point[0], point[1]]
                grid_copy[point[0], point[1]] = 2
                if value == 0: break
                if value == 0.5: new_info+=1

        if visualize:
            plt.figure()
            plt.imshow(grid_copy)
            plt.show()   
        return new_info
