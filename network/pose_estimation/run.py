import argparse
import logging
import sys
import time
import math
import cv2

import numpy as np
from .tf_pose.estimator import TfPoseEstimator
from .tf_pose.networks import get_graph_path, model_wh
#logger = logging.getLogger('TfPoseEstimatorRun')
#logger.handlers.clear()
#logger.setLevel(logging.DEBUG)
#ch = logging.StreamHandler()
#ch.setLevel(logging.DEBUG)
#formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
#ch.setFormatter(formatter)
#logger.addHandler(ch)


def open_pose_img(user_frame, image_url):
    w, h = model_wh("432x368")
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(w, h))
    # Convert the bytes data to a NumPy array
    frame_np = np.frombuffer(user_frame, dtype=np.uint8)
    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

    t = time.time()
    humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=4)
    elapsed = time.time() - t
    image_with_humans = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
    cv2.putText(image_with_humans,
                "FPS: %f" % (1.0 / (time.time())),
                (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)

    ret, jpeg = cv2.imencode('.jpg', image_with_humans)
    frame_array = np.frombuffer(jpeg, dtype=np.uint8)
    # Optionally, decode the NumPy array back to an image for further processing
    decoded_frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    return decoded_frame
    left_shoulder = None
    left_elbow = None
    keypoints = {}
    for human in humans:
        for k, v in human.body_parts.items():
            if k == 5:  # Left shoulder
                left_shoulder = (int(v.x * image_with_humans.shape[1]), int(v.y * image_with_humans.shape[0]))
            elif k == 7:  # Left elbow
                left_elbow = (int(v.x * image_with_humans.shape[1]), int(v.y * image_with_humans.shape[0]))

    if left_shoulder is not None and left_elbow is not None:
        x_midpoint = (left_shoulder[0] + left_elbow[0]) // 2
        y_midpoint = (left_shoulder[1] + left_elbow[1]) // 2

        # Load the image_url
        image_overlay = cv2.imread(image_url, cv2.IMREAD_UNCHANGED)
        # Calculate the Euclidean distance
        distance = math.sqrt((left_shoulder[0] - left_elbow[0]) ** 2 + (left_shoulder[1] - left_elbow[1]) ** 2)
        # Resize the image_url to fit within the frame
        image_overlay = cv2.resize(image_overlay, (distance, 50))
        # Overlay the image_url onto the frame at the calculated midpoint
        y_start = y_midpoint - image_overlay.shape[0] // 2
        y_end = y_start + image_overlay.shape[0]
        x_start = x_midpoint - image_overlay.shape[1] // 2
        x_end = x_start + image_overlay.shape[1]

        # Blend the image_url with the frame
        image_with_humans[y_start:y_end, x_start:x_end] = blend_transparent(image_with_humans[y_start:y_end, x_start:x_end], image_overlay)
        ret, jpeg = cv2.imencode('.jpg', image_with_humans)
        frame_array = np.frombuffer(jpeg, dtype=np.uint8)
        # Optionally, decode the NumPy array back to an image for further processing
        decoded_frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    else:
        ret, jpeg = cv2.imencode('.jpg', image_with_humans)
        frame_array = np.frombuffer(jpeg, dtype=np.uint8)
        # Optionally, decode the NumPy array back to an image for further processing
        decoded_frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    return decoded_frame


def blend_transparent(background, overlay):
    # Split out the transparency mask from the color info
    overlay_img = overlay[..., :3]  # Grab the BRG planes
    overlay_mask = overlay[..., 3:]  # And the alpha plane

    # Calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Blend the image and the overlay
    blended_part = (overlay_img * (overlay_mask / 255.0)) + (background * (background_mask / 255.0))

    # Round the values to integers
    blended_part = blended_part.astype(np.uint8)

    # Combine the two images
    return blended_part
