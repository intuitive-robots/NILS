import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import torch
from sklearn.cluster import DBSCAN

from nils.specialist_models.detectors.utils import create_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Metric3D:
    def __init__(self, model_type = "metric3d_vit_small"):
        self.model = torch.hub.load("yvanyin/metric3d", model_type,pretrain = True)
        self.model.to(device).eval()

        self.bsz = 4

    def predict(self, data,intrinsics):

        batched_images = create_batches(self.bsz, data)

        depths = []
        normals = []
        normal_confidences = []

        for batch in batched_images:

            data = batch

            data_processed, pad_info, intrinsic = self.prepare_images(data, intrinsics)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    pred_depth, confidence, output_dict = self.model.inference({'input': data_processed.to(device)})

            pred_depth = pred_depth.cpu()
            pred_depth = pred_depth.squeeze(1)
            pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1],
                         pad_info[2]: pred_depth.shape[1] - pad_info[3]]
            pred_depth = torch.nn.functional.interpolate(pred_depth[:,None, :, :], data[0].shape[:2],
                                                         mode='bilinear').squeeze(1)

            canonical_to_real_scale = intrinsic[0] / 1000.0  # 1000.0 is the focal length of canonical camera
            pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
            pred_depth = torch.clamp(pred_depth, 0, 300)

            pred_depth = pred_depth.cpu()

            pred_normal = output_dict['prediction_normal'][:, :3, :, :]
            normal_confidence = output_dict['prediction_normal'][:, 3, :,
                                :]  # see https://arxiv.org/abs/2109.09881 for details
            # un pad and resize to some size if needed

            # pred_normal = pred_normal[:, pad_info[0]: pred_normal.shape[1] - pad_info[1],
            #               pad_info[2]: pred_normal.shape[2] - pad_info[3]]
            # #pred_normal = pred_normal.squeeze()


            pred_normal = pred_normal.squeeze(1)
            pred_normal = pred_normal[:,:, pad_info[0]: pred_normal.shape[2] - pad_info[1],
                          pad_info[2]: pred_normal.shape[3] - pad_info[3]]
            pred_normal = torch.nn.functional.interpolate(pred_normal, data[0].shape[:2],
                                                          mode='bilinear').squeeze(1)


            pred_normal = pred_normal.permute(0,2, 3,1).cpu()

            #invert normals pointing in wrong direction
            up = np.array([0, 1, 0])

            inverted_normals = np.sum(np.array(pred_normal) * up, axis=-1) < 0

            pred_normal[inverted_normals] = -pred_normal[inverted_normals]


            valid_masks = get_valid_normal_masks(pred_normal.numpy())

            vis_normals = vis_surface_normal(pred_normal[0])

            vis_normals_mask = vis_normals[:,:,0] > 80

            depths.append(pred_depth)
            normals.append(pred_normal)
            normal_confidences.append(normal_confidence)

            del pred_depth, pred_normal, normal_confidence, output_dict






        # you can now do anything with the normal
        # such as visualize pred_normal
        #pred_normal_vis = pred_normal.cpu().numpy().transpose((1, 2, 0))
        #pred_normal_vis = (pred_normal_vis + 1) / 2

        depths = torch.cat(depths, dim=0)
        normals = torch.cat(normals, dim=0)
        normal_confidences = torch.cat(normal_confidences, dim=0)


        torch.cuda.empty_cache()


        return depths, normals, normal_confidences


    def prepare_images(self, images, intrinsic):
        gt_depth_scale = 200
        input_size = (616, 1064)  # for vit model




        h, w = images[0].shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        scaled_images  = [cv2.resize(image, (int(scale * w), int(scale * h))) for image in images]

        intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
        padding = [123.675, 116.28, 103.53]

        h, w = scaled_images[0].shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = [cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding) for image in scaled_images]
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        rgb = np.stack(rgb, axis=0)

        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((0,3, 1, 2))).float()
        rgb = torch.div((rgb - mean), std)


        return rgb, pad_info, intrinsic


def cluster_normals(normals):

    dbscan = DBSCAN(eps=0.1, min_samples=10)

    normals_flat = normals.reshape(-1, normals.shape[-1])
    dbscan.fit(normals_flat)
    labels = dbscan.labels_
    labels = labels.reshape(normals.shape[1], normals.shape[2])

    return labels


def calculate_angles(normals,up_vector=np.array([0, 1, 0])):
    """
    Calculate the angles between the normals and the up vector
    :param normals: the normal vectors
    :param up_vector: the up vector
    :return: the angles between the normals and the up vector
    """
    up_vector = up_vector / np.linalg.norm(up_vector)
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    dot_products = np.dot(normals, up_vector)
    angles_radians = np.arccos(
        np.clip(np.abs(dot_products), -1.0, 1.0))  # Use absolute value to handle opposite direction
    angles_degrees = np.degrees(angles_radians)
    return angles_degrees


def vis_surface_normal(normal: torch.tensor, mask: torch.tensor=None) -> np.array:
    """
    Visualize surface normal. Transfer surface normal value from [-1, 1] to [0, 255]
    Aargs:
        normal (torch.tensor, [h, w, 3]): surface normal
        mask (torch.tensor, [h, w]): valid masks
    """
    normal = normal.cpu().numpy().squeeze()
    n_img_L2 = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    n_img_norm = normal / (n_img_L2 + 1e-8)
    normal_vis = n_img_norm * 127
    normal_vis += 128
    normal_vis = normal_vis.astype(np.uint8)
    if mask is not None:
        mask = mask.cpu().numpy().squeeze()
        normal_vis[~mask] = 0
    return normal_vis



def get_valid_normal_masks(normals,up_vector=np.array([0, 1, 0]),threshold=40):
    """
    Get the valid normal masks
    :param normals: the normal vectors
    :param up_vector: the up vector
    :param threshold: the threshold of the angle between the normals and the up vector
    :return: the valid normal masks
    """

    valid_masks = []
    for normal in normals:
        normals_flat = normal.reshape(-1,normal.shape[-1])
        angles = calculate_angles(normals_flat,up_vector)

        valid_normals = angles < threshold

        valid_normals_map = valid_normals.reshape(normals.shape[1], normals.shape[2])

        valid_masks.append(valid_normals_map)

    #combine masks:
    valid_masks = np.stack(valid_masks, axis=0)
    valid_mask = np.logical_or.reduce(valid_masks, axis=0)


    return valid_mask