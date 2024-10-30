import cv2
import torch
import pybullet as p
import numpy as np
from sklearn.linear_model import RANSACRegressor

links_to_use = [0, 1, 2, 3, 4, 5, 6, 7]


# https://github.com/mees/hulc2/blob/56e51106a84080a93a12bdf232ca6fbb4303f01a/hulc2/affordance/dataset_creation/data_labeler.py#L313
def get_view_matrix_gripper(camera, robot_obs, point):
    pt, orn = robot_obs[:3], robot_obs[3:6]
    mode = "simulation"
    if "real_world" in mode:
        orn = p.getQuaternionFromEuler(orn)
        transform_matrix = np.reshape(p.getMatrixFromQuaternion(orn), (3, 3))
        transform_matrix = np.vstack([transform_matrix, np.zeros(3)])
        tcp2global = np.hstack([transform_matrix, np.expand_dims(np.array([*pt, 1]), 0).T])
        global2tcp = np.linalg.inv(tcp2global)
        point = global2tcp @ np.array([*point, 1])
        point = point[:3]
    else:
        orn = p.getQuaternionFromEuler(orn)
        tcp2cam_pos, tcp2cam_orn = camera.tcp2cam_T
        # cam2tcp_pos = [0.1, 0, -0.1]
        # cam2tcp_orn = [0.430235, 0.4256151, 0.559869, 0.5659467]
        cam_pos, cam_orn = p.multiplyTransforms(pt, orn, tcp2cam_pos, tcp2cam_orn)

        # Create projection and view matrix
        cam_rot = p.getMatrixFromQuaternion(cam_orn)
        cam_rot = np.array(cam_rot).reshape(3, 3)
        cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]

        # Extrinsics change as robot moves
        viewMatrix = p.computeViewMatrix(cam_pos, cam_pos + cam_rot_y, -cam_rot_z)
    return viewMatrix


def get_tcp_2_gripperjoints_transform(robot_uid, link_ids):

    pass



def tcp_to_gripper_handle(tcp):
    pt, orn = tcp[:3], tcp[3:6]
    orn_q = p.getQuaternionFromEuler(orn)
    transform_matrix = np.reshape(p.getMatrixFromQuaternion(orn_q), (3, 3))

    up = transform_matrix.T @ np.array([0, 0, 1])

    up = up / np.linalg.norm(up)

    translated = pt - (up * 0.08)

    return translated



def get_projection_from_gripper_locations(gripper_locations_2d,gripper_locations_3d):


    gripper_locations_3d_homogeneous = np.concatenate([gripper_locations_3d, np.ones((gripper_locations_3d.shape[0], 1))], axis=-1)
    gripper_locations_2d_homogeneous = np.concatenate([gripper_locations_2d, np.ones((gripper_locations_2d.shape[0], 1))], axis=-1)

    #solve pnp for camera pose


    reg_3d_2d = RANSACRegressor(random_state=0).fit(gripper_locations_3d_homogeneous, gripper_locations_2d_homogeneous)
    reg_2d_3d = RANSACRegressor(random_state=0).fit(gripper_locations_2d_homogeneous, gripper_locations_3d_homogeneous)



    n_inliers = reg_3d_2d.inlier_mask_.sum()

    projection_matrix_3d_2d = reg_3d_2d.estimator_.coef_
    projection_matrix_3d_2d[:,-1] = reg_3d_2d.estimator_.intercept_

    projection_matrix_2d_3d = reg_2d_3d.estimator_.coef_
    projection_matrix_2d_3d[:,-1] = reg_2d_3d.estimator_.intercept_

    pred_points = reg_3d_2d.predict(gripper_locations_3d_homogeneous)[:, :-1].astype(int)

    return projection_matrix_3d_2d, n_inliers, pred_points

