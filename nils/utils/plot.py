import math

import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from moviepy.config import change_settings
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import TextClip, VideoClip
from PIL import Image, ImageDraw
from tqdm import tqdm

from nils.specialist_models.detectors.utils import plot_boxes_np_sv

change_settings({"IMAGEMAGICK_BINARY": "/home/i53/mitarbeiter/reuss/imagemick/magick"})

def plot_images(images, annotation, end_frame_idx=-1):
    fig = plt.figure()
    f, axarr = plt.subplots(1, 2, layout='constrained')

    if len(images.shape) > 4:
        axarr[0].imshow(images[-1, 0, ...])
        axarr[1].imshow(images[-1, 1, ...])
    else:
        axarr[0].imshow(images[0, ...])
        axarr[1].imshow(images[end_frame_idx, ...])

    axarr[0].set_title("Start Frame")
    axarr[1].set_title("End Frame")
    plt.suptitle(annotation)
    for ax in axarr.ravel():
        ax.set_axis_off()
    fig.tight_layout()
    plt.show()


def plot_image(image):
    #plt.figure(figsize=(15,15))
    plt.imshow(image)
    plt.axis('off')
    #plt.tight_layout()
    plt.show()






def plot_image_sequence(images, common_title=None, rows = 1,cmap='gray', figsize=(15, 5)):
    """Plot a sequence of images in one row using matplotlib with a common title.

    Parameters:
    - images: List of image arrays (numpy arrays).
    - common_title: Common title for all subfigures. If None, no common title will be displayed.
    - cmap: Colormap to use for displaying images. Default is 'gray'.
    - figsize: Tuple specifying the size of the figure. Default is (15, 5).
    """
    num_images = len(images)
    nrows = rows
    figsize =(num_images * 4, 4 * nrows)
    # Create a new figure with the specified size
    plt.figure(figsize=figsize)

    # Display common title above subplots if provided
    if common_title is not None:
        plt.suptitle(common_title, fontsize=16)


    # Plot each image in the row
    for i in range(num_images):

        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.axis('off')  # Turn off axis labels

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()  # Leave space at the top for the common title
    #plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                       hspace=0, wspace=0)
    #plt.margins(0, 0)
    # Show the plot
    plt.show()


def plot_multiple_goals(images, annotation, end_frame_idxs=[-1]):

    ncols = min(5,images.shape[0])
    fig = plt.figure(figsize=(15, 15//ncols))
    n_rows = len(end_frame_idxs) // ncols + (len(end_frame_idxs) % ncols > 0)
    images = np.stack(images)
    for i, idx in enumerate(end_frame_idxs):
        ax = fig.add_subplot(n_rows, ncols, i + 1)

        if len(images.shape) > 4:
            ax.imshow(images[-1, idx, ...])
        else:
            ax.imshow(images[idx, ...])

        if i == 0:

            ax.set_title(str(i + 1) + ":" + annotation)
        else:
            ax.set_title(str(i + 1))
        ax.set_axis_off()
        fig.tight_layout()
    plt.show()




import cv2
import numpy as np


def pad_image(image,padding):
    return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

def create_video(frames, keyframe_ranges, keyframe_scores, nlp_descriptions,out_path = 'output_video.mp4', target_res = 800):
    # Get the shape of the frames
    
    longest_side = max(frames[0].shape)
    scale = target_res / longest_side
    

    height,width = frames[0].shape[0] * scale, frames[0].shape[1] * scale
    height, width, layers = int(height), int(width), frames[0].shape[2]
    
    print(height,width,layers)

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(out_path, fourcc, 5.0, (width, height + 100))

    for i in range(len(frames)):
        frame = cv2.cvtColor(np.array(frames[i]), cv2.COLOR_BGR2RGB) 
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
      
        # Create an image with extra space at the bottom
        frame_with_text_space = np.full((height + 100, width, layers), 255, dtype=np.uint8) 
        frame_with_text_space[:height, :width] = frame
        
        pil_img = Image.fromarray(frame_with_text_space)
        draw = ImageDraw.Draw(pil_img)


        
        for j in range(len(keyframe_ranges)):
            start, end = keyframe_ranges[j]
            if start <= i <= end:
                state = nlp_descriptions[j]
                score = keyframe_scores[j]
                # Put state, score and description on the frame
                cv2.putText(frame_with_text_space, 'Action: ' + str(state), (50, height + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame_with_text_space, 'Score: ' + str(score), (50, height + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
        # Write the frame into the file 'output_video.mp4'
        video.write(frame_with_text_space)

    # Release everything after the job is finished
    video.release()
    cv2.destroyAllWindows()






def save_video(observations, vid_dir="video.avi", annotate=True):
    frames = observations
    height, width, channels = frames[0].shape

    fps = 8
    if annotate:
        # video = cv2.VideoWriter(vid_dir, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))

        writer = FFMpegWriter(fps=fps)
        frame = frames[0]
        fig, ax = plt.subplots()
        ax.imshow(frame)
        plt.axis("off")

        fig.suptitle(observations["annotations"][0])

        def animate(i):
            ax.clear()
            frame = frames[i]
            ax.imshow(frame)
            plt.axis("off")

            fig.suptitle(observations["annotations"][i])

        anim = FuncAnimation(fig, animate, frames=len(frames))
        anim.save(vid_dir, writer=writer)
    else:
        video = cv2.VideoWriter(vid_dir, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
        for frame in list(frames):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            video.write(np.array(frame))
        cv2.destroyAllWindows()
        video.release()


def draw_arrow_on_image(image_np, start_point, end_point, color=(0, 0, 255), thickness=4, alpha=0.7):
    """Draw a red arrow on the given image.

    Parameters:
        image_np (numpy.ndarray): Input image represented as a NumPy array.
        start_point (tuple): Start point of the arrow (x, y).
        end_point (tuple): End point of the arrow (x, y).
        color (tuple): Color of the arrow (B, G, R). Default is red (0, 0, 255).
        thickness (int): Thickness of the arrow. Default is 2.

    Returns:
        numpy.ndarray: Image with the arrow drawn on it.
    """
    # Convert the NumPy array image to OpenCV format

    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Draw the arrow on the image

    angle = math.atan2(start_point[1] - end_point[1], start_point[0] - end_point[0])
    arrowhead_length = 20
    arrowhead_end1 = (int(end_point[0] + arrowhead_length * math.cos(angle + math.pi / 4)),
                      int(end_point[1] + arrowhead_length * math.sin(angle + math.pi / 4)))
    arrowhead_end2 = (int(end_point[0] + arrowhead_length * math.cos(angle - math.pi / 4)),
                      int(end_point[1] + arrowhead_length * math.sin(angle - math.pi / 4)))

    image_cv = cv2.line(image_cv, start_point, end_point, color, thickness)
    pts = np.array([end_point, arrowhead_end1, arrowhead_end2])
    cv2.fillPoly(image_cv, [pts], color)

    # Convert the image back to RGB format
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    image = cv2.addWeighted(image_np, 1 - alpha, image_rgb, alpha, 0)

    return image


def draw_karaoke_text(frame, text, pos, font, font_scale, color, thickness, progress):
    """Draw text on the frame with part of it colored based on progress.

    Parameters:
        frame (numpy array): The frame on which to draw.
        text (str): The text to draw.
        pos (tuple): The position to draw the text (x, y).
        font (int): The font type.
        font_scale (float): The scale of the font.
        color (tuple): The color of the text (B, G, R).
        thickness (int): The thickness of the text.
        progress (float): The progress of the coloring (0 to 1).
    """
    # Calculate the width of the text
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the split position for coloring
    split_pos = int(len(text) * progress)

    # Split the text into colored and non-colored parts
    colored_text = text[:split_pos]
    non_colored_text = text[split_pos:]

    # Draw the colored part
    colored_text_size, _ = cv2.getTextSize(colored_text, font, font_scale, thickness)
    cv2.putText(frame, colored_text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

    # Calculate position for the non-colored text
    new_pos = (pos[0] + colored_text_size[0], pos[1])

    # Draw the non-colored part
    cv2.putText(frame, non_colored_text, new_pos, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
def draw_multiline_text(frame, text, pos, font, font_scale, color, thickness):
    """Draw multiline text on the frame.

    Parameters:
        frame (numpy array): The frame on which to draw.
        text (str): The text to draw.
        pos (tuple): The position to draw the text (x, y).
        font (int): The font type.
        font_scale (float): The scale of the font.
        color (tuple): The color of the text (B, G, R).
        thickness (int): The thickness of the text.
    """
    x, y = pos
    for i, line in enumerate(text.split('\n')):
        y_position = y + i * int(cv2.getTextSize(line, font, font_scale, thickness)[0][1] * 1.5)
        cv2.putText(frame, line, (x, y_position), font, font_scale, color, thickness, cv2.LINE_AA)


def create_annotated_video_matplotlib(annotations, frames, out_dir, fps=12):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Hide the axis
    ax.axis('off')

    # Create an image object on the axis, initialized with the first frame
    img = ax.imshow(frames[0])

    # Create a text object on the axis, initialized with the first annotation
    text = ax.text(0.5, 0.1, annotations[0][1], fontsize=16, ha='center', va='center', transform=fig.transFigure)
    
    # Create a progress bar object
    progress_bar = plt.Rectangle((0, 0.9), 0, 0.05, transform=fig.transFigure, color='blue')
    fig.patches.append(progress_bar)

    # Create circles for key state dots
    dots = []
    for end_frame_index, _ in annotations:
        dot_position = (end_frame_index / len(frames), 0.925)
        circle = plt.Circle(dot_position, 0.01, color='black', transform=fig.transFigure)
        dots.append(circle)
        fig.patches.append(circle)

    # Initialization function for the animation
    def init():
        return [img, text, progress_bar] + dots

    # Update function for the animation
    def update(i):
        # Update the image data
        img.set_array(frames[i])

        # Find the latest annotation for the current frame
        for end_frame_index, annotation_text in annotations:
            if i <= end_frame_index:
                text.set_text(annotation_text)
                break

        # Update the progress bar
        progress = i / len(frames)
        progress_bar.set_width(progress)

        # Change the color of the progress bar according to progress
        if progress < 0.5:
            progress_bar.set_color('red')
        elif progress < 0.75:
            progress_bar.set_color('yellow')
        else:
            progress_bar.set_color('green')

        return [img, text, progress_bar] + dots

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init, blit=True)

    # Save the animation as a video file
    ani.save(out_dir, writer='ffmpeg', fps=fps)

def create_annotated_video(annotations, frames, out_dir, ks_scale_factor=1, fps=12):
    height, width, _ = frames[0].shape
    box_height = 100  # Height of the white box for text
    new_height = height + box_height
    progress_bar_height = 20  # Height of the progress bar
    progress_bar_width = width - 20  # Width of the progress bar
    progress_bar_start = (10, new_height - 30)  # Start position of the progress bar

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(out_dir, fourcc, fps, (width, new_height))
    gif_frames = []

    for i, frame in enumerate(frames):
        #frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frame = np.array(frame)
        frame_with_box = np.ones((new_height, width, 3), dtype=np.uint8) * 255  # White background
        frame_with_box[:height] = frame  # Copy original frame into the new frame

        for j, (end_frame_index, text) in enumerate(annotations):
            if i <= end_frame_index:
                color = (0, 0, 0)  # Black text color
                draw_multiline_text(frame_with_box, text, (10, height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                break  # Only draw the latest relevant annotation

        # Draw the progress bar
        progress = i / len(frames)
        cv2.rectangle(frame_with_box, progress_bar_start, (progress_bar_start[0] + int(progress * progress_bar_width), progress_bar_start[1] + progress_bar_height), (0, 0, 255), -1)

        # Draw the round dots for key states
        for end_frame_index, _ in annotations:
            dot_position = (progress_bar_start[0] + int(end_frame_index / len(frames) * progress_bar_width), progress_bar_start[1] + progress_bar_height // 2)
            cv2.circle(frame_with_box, dot_position, 5, (0, 255, 0), -1)  # Green dot

        out.write(frame_with_box)
        gif_frames.append(frame_with_box)
    imageio.mimsave(out_dir +".gif", gif_frames, fps=fps)
    out.release()
    

    


def plot_trajectory(points,start_idx, end_idx):
    # Plot the points as a trajectory
    subset = end_idx
    start = start_idx
    plt.plot(points[start:subset, 0], 250 - points[start:subset, 1], marker='o', linestyle='-')
    for i, point in enumerate(points[start:subset]):
        plt.text(point[0], 250 - point[1], str(i), ha='right')

    plt.title('Trajectory of 2D Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 250)
    plt.ylim(0, 250)
    plt.grid(True)
    plt.show()

def crop_images_with_boxes(image, box,resize = False, resize_dim = 225, padding = 10):
    
    
    orig_shape = image.shape
    x1, y1, x2, y2 = box
    x1 = int(max(0,x1-padding))
    y1 = int(max(0,y1-padding))
    x2 = int(min(x2+padding,image.shape[1]))
    y2 = int(min(y2+padding,image.shape[0]))
    

    
    
    if image.dtype == "bool":
        image = image.astype(np.uint8)
    cropped_image = image[y1:y2,x1:x2]
    
    if resize:
        if resize_dim is not None:
            cropped_image = cv2.resize(np.array(cropped_image), (resize_dim, resize_dim),interpolation=cv2.INTER_CUBIC)
        else:
            cropped_image = cv2.resize(np.array(cropped_image), (orig_shape[1], orig_shape[0]),interpolation=cv2.INTER_CUBIC)
        
    return cropped_image


def create_image_prompt(filtered_objects, moved_object, obj_movement, scene_graphs_subset_prompt):
    if scene_graphs_subset_prompt[-1].get_node(moved_object).seg_mask.sum() > 0:
        annotated_image = scene_graphs_subset_prompt[-1].annotate_image(moved_object,
                                                                        filtered_objects)
        # annotated_image = create_visualization(last_img_np,moved_object_seg_mask,moved_object)

        start_point = np.mean(
            np.where(scene_graphs_subset_prompt[0].get_node(moved_object).seg_mask),
            axis=1).astype(int)[::-1]
        end_point = np.mean(
            np.where(scene_graphs_subset_prompt[-1].get_node(moved_object).seg_mask),
            axis=1).astype(int)[::-1]
        annotated_image = draw_arrow_on_image(annotated_image, start_point=(
                start_point - 20 * obj_movement / np.linalg.norm(obj_movement)).astype(int),
                                              end_point=end_point)
    else:
        annotated_image = None

    return annotated_image


def annotate_frame(image, object_stats_frame=None):
    all_object_stats = object_stats_frame

    detections = np.array([all_object_stats[key]["box"] for key in all_object_stats.keys() if
                           all_object_stats[key]["score"] is not None and not np.isnan(
                               all_object_stats[key]["score"])])
    labels = np.array([key for key in all_object_stats.keys() if
                       all_object_stats[key]["score"] is not None and not np.isnan(all_object_stats[key]["score"])])
    scores = np.array([all_object_stats[key]["score"] for key in all_object_stats.keys() if
                       all_object_stats[key]["score"] is not None and not np.isnan(all_object_stats[key]["score"])])

    image = np.array(image)
    annotated_img = plot_boxes_np_sv(image, detections, labels, scores, return_image=True)

    return annotated_img
