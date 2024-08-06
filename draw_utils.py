import logging
import cv2

import numpy as np

logger = logging.getLogger()

BONES = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
         [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
         [0,15], [15,17]]

JOINT_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

BONE_COLORS = [[153, 0, 0], [153, 51, 0], [153, 102, 0], [153, 153, 0], [102, 153, 0], [51, 153, 0], [0, 153, 0], [0, 153, 51],
               [0, 153, 102], [0, 153, 153], [0, 102, 153], [0, 51, 153], [0, 0, 153], [51, 0, 153], [102, 0, 153],
               [153, 0, 153], [153, 0, 102]]

KEYPOINT_INFO = {
    0: dict(name='nose'),
    1: dict(name='neck'),
    2: dict(name='right_shoulder'),
    3: dict(name='right_elbow'),
    4: dict(name='right_wrist'),
    5: dict(name='left_shoulder'),
    6: dict(name='left_elbow'),
    7: dict(name='left_wrist'),
    8: dict(name='right_hip'),
    9: dict(name='right_knee'),
    10: dict(name='right_ankle'),
    11: dict(name='left_hip'),
    12: dict(name='left_knee'),
    13: dict(name='left_ankle'),
    14: dict(name='right_eye'),
    15: dict(name='left_eye'),
    16: dict(name='right_ear'),
    17: dict(name='left_ear'),
}

UPPER_BONES = [[2, 3], [3, 4], [5, 6], [6, 7], [2, 5], [5, 11], [11, 8], [8, 2]]
RADIUS_LINK = [2, 5]
KEEP_LINKS = [[3, 4], [6, 7]]
TORSO = [2, 5, 11, 8]

MASK_COLOR = [255, 255, 255]

mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]


def cords_to_map(cords, img_size, affine_matrix=None, sigma=6):
    cords = cords.astype(float)
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == -1 or point[1] == -1:
            continue
        if affine_matrix is not None:
            point_ =np.dot(affine_matrix, np.matrix([point[1], point[0], 1]).reshape(3,1))
            point_0 = int(point_[1])
            point_1 = int(point_[0])
        else:
            point_0 = int(point[0])
            point_1 = int(point[1])
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point_0) ** 2 + (xx - point_1) ** 2) / (2 * sigma ** 2))
    return result


def draw_pose_from_cords(array, img_size, radius=2, draw_bones=True, kpt_thr=0.2):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)

    if draw_bones:
        for i, (f, t) in enumerate(BONES):
            from_missing = array[f][2] < kpt_thr
            to_missing = array[t][2] < kpt_thr
            if from_missing or to_missing:
                continue
            cv2.line(colors, (int(array[f][1]), int(array[f][0])),
                     (int(array[t][1]), int(array[t][0])), BONE_COLORS[i], radius, cv2.LINE_AA)

    for i, joint in enumerate(array):
        if array[i][2] < kpt_thr:
            continue
        cv2.circle(colors, (int(joint[1]), int(joint[0])), radius + 1, JOINT_COLORS[i], -1, cv2.LINE_AA)

    return colors


def mmpose_to_coco(keypoints_info, kpt_thr=0.2):
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    neck[:, 2:3] = np.logical_and(
        keypoints_info[:, 5, 2:3] > kpt_thr,
        keypoints_info[:, 6, 2:3] > kpt_thr).astype(int)
    new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
    new_keypoints_info[:, openpose_idx] = \
        new_keypoints_info[:, mmpose_idx]
    new_keypoints_info[0, :, :-1][new_keypoints_info[0, :, :-1] < 0] = 0
    out = np.concatenate([new_keypoints_info[0, :, :-1][:, ::-1], np.expand_dims(new_keypoints_info[0, :, -1], 1)], axis = 1)
    return out


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette
