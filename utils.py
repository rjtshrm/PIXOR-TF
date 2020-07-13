import numpy as np
#import cv2
import time
from shapely.geometry import Polygon
cv2=None
from matplotlib import pyplot as plt


SAVE_DIR='./plots/'

def rotation_z(yaw):
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ], dtype='float32')

def rotation_y(pitch):
    return np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ], dtype='float32')


def rotation_x(roll):
    return np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ], dtype='float32')



def iou_2d(bbx1, bbx2):
    '''
    iou_2d: calculates intersection / union area of two shapes
    :param bbx1: [(top left coords), (bottom_right_cords)]
    :param bbx2: [(top left coords), (bottom_right_cords)]
    :return: iou
    '''
    x1 = max(bbx1[0, 0], bbx2[0, 0])
    y1 = max(bbx1[0, 1], bbx2[0, 1])
    x2 = min(bbx1[1, 0], bbx2[1, 0])
    y2 = min(bbx1[1, 1], bbx2[1, 1])

    intersection_area = max((x2 - x1), 0) * max((y2 - y1), 0)
    area_bbx1 = abs((bbx1[1, 0] - bbx1[0, 0]) * (bbx1[1, 1] - bbx1[0, 1]))
    area_bbx2 = abs((bbx2[1, 0] - bbx2[0, 0]) * (bbx2[1, 1] - bbx2[0, 1]))

    union_area = area_bbx1 + area_bbx2 - intersection_area
    union_area = union_area + 1e-6   # to avoid zero division
    iou = intersection_area / union_area
    return iou

def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)

def convert_format(boxes_array):
    """
    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """

    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)

def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    return an numpy array of the positions of picks
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    polygons = convert_format(boxes)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)


def filter_pred(pred, objectness_threshold, iou_threshold, norm):
    from kitti import KITTI
    cls_pred = pred[..., 0]
    loc_grid = pred[..., 1:]
    decoded_corners = KITTI.decode_labels(loc_grid, de_normalize=norm)
    activation = cls_pred > objectness_threshold
    num_boxes = int(activation.sum())

    if num_boxes == 0:
        print("No bounding box found")
        return np.array([]), np.array([])

    corners = decoded_corners[activation].reshape(-1, 4, 2)
    scores = cls_pred[activation]

    # NMS
    selected_ids = non_max_suppression(corners, scores, iou_threshold)
    corners = corners[selected_ids]
    scores = scores[selected_ids]

    return corners, scores

def extract_2d_vertices_from_anchors(theta, x_center, y_center, l, w):
    """
        Return 2d Vertices of bev bbx
    """
    cost = np.cos(theta)
    sint = np.sin(theta)
    front_right_x = np.expand_dims(x_center + l / 2 * cost + w / 2 * sint, axis=-1)
    front_right_y = np.expand_dims(y_center + l / 2 * sint - w / 2 * cost, axis=-1)

    front_left_x = np.expand_dims(x_center + l / 2 * cost - w / 2 * sint, axis=-1)
    front_left_y = np.expand_dims(y_center + l / 2 * sint + w / 2 * cost, axis=-1)

    rear_right_x = np.expand_dims(x_center - l / 2 * cost + w / 2 * sint, axis=-1)
    rear_right_y = np.expand_dims(y_center - l / 2 * sint - w / 2 * cost, axis=-1)

    rear_left_x = np.expand_dims(x_center - l / 2 * cost - w / 2 * sint, axis=-1)
    rear_left_y = np.expand_dims(y_center - l / 2 * sint + w / 2 * cost, axis=-1)

    return np.concatenate((rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                           front_right_x, front_right_y, front_left_x, front_left_y), axis=-1)

def postprocess_results(ip_pc, op, true_op, raw_lidar, objectness_threshold=0.4, iou_threshold=0.3, visualization=False, norm=False):
    pcorners, pscores  = filter_pred(op, objectness_threshold, iou_threshold, norm)

    gcorners, gscores = None, None
    if true_op is not None:
        gcorners, gscores = filter_pred(true_op, 0, 0, norm)

    x = raw_lidar[:, 0]
    y = raw_lidar[:, 1]

    fig, axs = plt.subplots(1)
    axs.scatter(x, y, s=0.1)

    connected_vertices = [[0, 1], [1, 2], [2, 3], [3, 0]]

    if visualization:
        for corners in pcorners:
            corners = corners.transpose()
            plt.plot(*corners[:, connected_vertices], color='red')

        for corners in gcorners:
            corners = corners.transpose()
            plt.plot(*corners[:, connected_vertices], color='green')

        #plt.savefig('results/{0}.png'.format(time.time()))
        plt.show()

    return pcorners, pscores, gcorners, gscores


def MAP(pscores, gscores):
    # TO DO
    pass
