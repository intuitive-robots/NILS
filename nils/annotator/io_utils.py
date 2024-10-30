import os
import pickle

import cv2
import numpy as np

from nils.utils.plot import annotate_frame


def save_annotated_frames(object_manager, depth_predictions, batch, path, save_video=True):
    if not os.path.exists(path):
        os.makedirs(path)

    # all_object_stats = self.object_manager.get_all_object_stats()

    orig_frames = os.path.join(path, "original")
    os.makedirs(orig_frames, exist_ok=True)

    robot_save_path = os.path.join(path, "robot")
    os.makedirs(robot_save_path, exist_ok=True)

    depth_dir = os.path.join(path, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    if save_video:
        out_dir_orig = os.path.join(orig_frames, "original.mp4")
        writer_orig = cv2.VideoWriter(out_dir_orig, cv2.VideoWriter_fourcc(*"mp4v"), 8,
                                      (batch["rgb_static"][0].shape[1], batch["rgb_static"][0].shape[0]))

        out_dir_robot = os.path.join(robot_save_path, "robot.mp4")
        writer_robot = cv2.VideoWriter(out_dir_robot, cv2.VideoWriter_fourcc(*"mp4v"), 8,
                                       (object_manager.robot.mask[0].shape[1],
                                        object_manager.robot.mask[0].shape[0]))

        out_dir_depth = os.path.join(depth_dir, "depth.mp4")
        writer_depth = cv2.VideoWriter(out_dir_depth, cv2.VideoWriter_fourcc(*"mp4v"), 8,
                                       (depth_predictions[0].shape[1], depth_predictions[0].shape[0]))

        annotated_image = annotate_frame(batch["rgb_static"][0], object_manager.get_object_stats_frame(0))
        out_dir_annotated = os.path.join(path, "annotated.mp4")
        writer_annotated = cv2.VideoWriter(out_dir_annotated, cv2.VideoWriter_fourcc(*"mp4v"), 8,
                                           (annotated_image.shape[1], annotated_image.shape[0]))

    for i in range(len(batch["rgb_static"])):

        object_stats_frame = object_manager.get_object_stats_frame(i)

        robot_info = object_manager.get_robot_stats_frame(i)

        object_stats_frame["robot"] = robot_info["robot"]

        img = annotate_frame(batch["rgb_static"][i], object_stats_frame)

        depth = np.log(depth_predictions[i])
        robot_mask = object_manager.robot.mask[i]

        robot_frame_name = f"{batch['frame_name'][i]}_robot.jpg" if "frame_name" in batch.keys() else f"{i}_robot.jpg"

        orig_frame_cv2 = cv2.cvtColor(np.array(batch["rgb_static"][i]), cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(os.path.join(orig_frames, f"{i}.jpg"), orig_frame_cv2)

        robot_mask_int = robot_mask.astype(np.uint8) * 255
        success = cv2.imwrite(os.path.join(robot_save_path, robot_frame_name), robot_mask_int)

        frame_name = f"{batch['frame_name'][i]}_annotated.jpg" if "frame_name" in batch.keys() else f"{i}.jpg"
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        success = cv2.imwrite(os.path.join(path, frame_name), img)

        depth_frame_name = f"{batch['frame_name'][i]}_depth.jpg" if "frame_name" in batch.keys() else f"{i}_depth.jpg"
        depth_img = depth
        # to grayscale:
        depth_img = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img))

        depth_img_frame = (depth_img * 255).astype(np.uint8)
        success = cv2.imwrite(os.path.join(depth_dir, depth_frame_name), depth_img_frame)

        if save_video:
            writer_orig.write(orig_frame_cv2)
            writer_robot.write(cv2.cvtColor(robot_mask_int, cv2.COLOR_GRAY2BGR))
            writer_depth.write(cv2.cvtColor(depth_img_frame, cv2.COLOR_GRAY2BGR))
            writer_annotated.write(img)

        if not success:
            print("Failed to write image")

            continue
    if save_video:
        writer_orig.release()
        writer_robot.release()
        writer_depth.release()
        writer_annotated.release()


def save_data_cache(object_manager, flow_predictions, objectness, region_proposals, path):
    data_cache = {"object_manager": object_manager, "flow_predictions": flow_predictions,
                  "objectness": objectness, "region_proposals": region_proposals, }

    with open(path, "wb") as f:
        pickle.dump(data_cache, f)
