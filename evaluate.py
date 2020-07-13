from datagen import  DataGen
from tf_model import pixor_modified
import numpy as np
import tensorflow as tf
from loss import custom_loss, class_loss, reg_loss
import utils
from tensorflow.keras import optimizers
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--objectness_threshold', default=0.1, type=float)
parser.add_argument('--iou_threshold', default=0.01, type=float)
parser.add_argument('--visualization', default=True, type=bool)
parser.add_argument('--val', default=True, type=bool)
parser.add_argument('--norm', default=False, type=bool)
FLAGS = parser.parse_args()



model = pixor_modified(input_shape=(800, 700, 36), num_block=[3, 6, 6, 3], use_bn=True, use_height=False)

TEST_KITTI_PATH_VELODYNE = '/mnt/raid_data/srr7rng/KITTI/data_object_velodyne/validation/velodyne/'
TEST_KITTI_PATH_LABELS = '/mnt/raid_data/srr7rng/KITTI/data_object_label_2/validation/label_2/'
TEST_KITTI_PATH_CALIBS = '/mnt/raid_data/srr7rng/KITTI/data_object_calib/validation/calib/'

log_dir = 'logs/pixor'
checkpoint_dir = 'checkpoint'


testdatagenerator = DataGen(TEST_KITTI_PATH_VELODYNE, TEST_KITTI_PATH_LABELS, TEST_KITTI_PATH_CALIBS , batch_size=5, type='val', use_cache=True, augmentation=False, use_height=False, raw_lidar=True, norm=FLAGS.norm)

cps = tf.train.latest_checkpoint(checkpoint_dir)
print(f'latest checkpoint: {cps}')
model.load_weights(cps)


for i in range(len(testdatagenerator)):
    ip, true_op, pcd = testdatagenerator.__getitem__(0)

    op = model.predict(ip)

    pred_scores_len = []
    gt_score_len = []

    for per_btch_idx in range(ip.shape[0]):
        print(per_btch_idx, ip.shape)
        pcorners, pscores, gcorners, gscores = utils.postprocess_results(ip[per_btch_idx], op[per_btch_idx, ...], true_op[per_btch_idx, ...] if true_op is not None else None,
                                  pcd[per_btch_idx], FLAGS.objectness_threshold, FLAGS.iou_threshold, FLAGS.visualization)
    break