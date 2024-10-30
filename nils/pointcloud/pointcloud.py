import torch

from nils.pointcloud.pointcloud_utils import *

IN_CONTACT_DISTANCE = 0.01
CLOSE_DISTANCE = 0.17
INSIDE_THRESH = 0.02
ON_TOP_OF_THRESH = 0.5
NORM_THRESH_FRONT_BACK = 0.9
NORM_THRESH_UP_DOWN = 0.85
NORM_THRESH_LEFT_RIGHT = 0.85
OCCLUDE_RATIO_THRESH = 0.85
DEPTH_THRESH = 0.9
IS_NOISE_THRESH = 1.1
DEPTH_THRESH = 0.9



IN_CONTACT_DISTANCE_3D = 0.01
CLOSE_DISTANCE_3D = 0.1
INSIDE_THRESH_3D = 0.02
ON_TOP_OF_THRESH_3D = 0.8
NORM_THRESH_FRONT_BACK_3D = 0.9
NORM_THRESH_UP_DOWN_3D = 0.85
NORM_THRESH_LEFT_RIGHT_3D = 0.85
OCCLUDE_RATIO_THRESH_3D = 0.85
DEPTH_THRESH_3D = 0.9
IS_NOISE_THRESH_3D = 1.1
DEPTH_THRESH_3D = 0.9

def get_obj_points(depth_map, seg_mask=None):

    #points = depth_frame_to_camera_space_xyz(depth_map, seg_mask,fov = 70)


    width = depth_map.shape[1]
    height = depth_map.shape[0]
    FY = depth_map.shape[0]  * 0.6
    FX = depth_map.shape[1] * 0.6

    focal_length_x, focal_length_y = (FX, FY)
    y,x = np.where(seg_mask)

    x = (x - width / 2) / focal_length_x
    y = (y - height / 2) / focal_length_y

    y = y*-1

    z = depth_map[seg_mask]
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)

    return points

    focal_length_x, focal_length_y = (FX, FY)

    return camera_space_xyz * depth_map[mask][None, :]


    return points


def apply_mask_to_image(image, mask):
    """
    Apply a binary mask to an image. The mask should be a binary array where the regions to keep are True.
    """
    masked_image = image.copy()

    masked_image[~mask] = 0
    return masked_image



def create_point_cloud_from_rgbd(rgb_image, depth_image, intrinsic_parameters):


    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_image),
            o3d.geometry.Image(depth_image),
            depth_scale=1.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False
        )
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(intrinsic_parameters['width'], intrinsic_parameters['height'],
                             intrinsic_parameters['fx'], intrinsic_parameters['fy'],
                             intrinsic_parameters['cx'], intrinsic_parameters['cy'])
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    return pcd

def get_obj_depth(seg_mask):
    raise NotImplementedError









