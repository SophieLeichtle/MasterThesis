import numpy as np
from igibson.utils.mesh_util import quat2rotmat, xyzw2wxyz
from transforms3d.euler import euler2quat

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def openglf_to_wf(env):
    eye_pos, eye_orn = env.robots[0].links["eyes"].get_position_orientation()
    camera_in_wf = quat2rotmat(xyzw2wxyz(eye_orn))
    camera_in_wf[:3,3] = eye_pos

    # Transforming coordinates of points from opengl frame to camera frame
    camera_in_openglf = quat2rotmat(euler2quat(np.pi / 2.0, 0, -np.pi / 2.0))

    # Pose of the simulated robot in world frame
    robot_pos, robot_orn = env.robots[0].get_position_orientation()
    robot_in_wf = quat2rotmat(xyzw2wxyz(robot_orn))
    robot_in_wf[:3, 3] = robot_pos

    # Pose of the camera in robot frame
    cam_in_robot_frame = np.dot(np.linalg.inv(robot_in_wf), camera_in_wf)

    return np.matmul(robot_in_wf, np.matmul(cam_in_robot_frame, camera_in_openglf))

def pixel_to_point(env, row, column, depth):
    z = depth * env.config["depth_high"]
    assert(z != 0)
    f = 579.4 # TODO get from intrinsics
    x = (column - 320) * z / f
    y = (240 - row) * z / f
    point_in_openglf = np.array([x,y,-z, 1])
    return np.dot(openglf_to_wf(env), point_in_openglf)[:3]