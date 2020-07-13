import os
import sys
import glob
import subprocess

if __name__ == '__main__':
    train_data_ratio = 80 # val = 100 - 80 
    KITTI_PATH_VELODYNE = '/mnt/raid_data/srr7rng/KITTI/data_object_velodyne'
    KITTI_PATH_LABELS = '/mnt/raid_data/srr7rng/KITTI/data_object_label_2'
    KITTI_PATH_CALIBS = '/mnt/raid_data/srr7rng/KITTI/data_object_calib'


    v_files = glob.glob(f'{KITTI_PATH_VELODYNE}/training/velodyne/*.bin')
    l_files = glob.glob(f'{KITTI_PATH_LABELS}/training/label_2/*.txt')
    c_files = glob.glob(f'{KITTI_PATH_CALIBS}/training/calib/*.txt')
    
    frame_ids = []
    for f in v_files:
        frame_id= f.split('/')[-1][:-4]
        frame_ids.append(frame_id)
    
    dataset_len = len(frame_ids)
    train_len = dataset_len * train_data_ratio // 100
    val_len = dataset_len - train_len

    print(f'Splitting data set (len={dataset_len}): train files={train_len}, validation files={val_len}')
    
    val_frame_id = frame_ids[train_len:]

    subprocess.Popen(['mkdir', '-p', f'{KITTI_PATH_VELODYNE}/validation/velodyne'])
    subprocess.Popen(['mkdir', '-p', f'{KITTI_PATH_LABELS}/validation/label_2'])
    subprocess.Popen(['mkdir', '-p', f'{KITTI_PATH_CALIBS}/validation/calib'])

    for f in val_frame_id:
        subprocess.Popen(['mv', f'{KITTI_PATH_VELODYNE}/training/velodyne/{f}.bin', f'{KITTI_PATH_VELODYNE}/validation/velodyne/.'])
        subprocess.Popen(['mv', f'{KITTI_PATH_LABELS}/training/label_2/{f}.txt', f'{KITTI_PATH_LABELS}/validation/label_2/.'])
        subprocess.Popen(['mv', f'{KITTI_PATH_CALIBS}/training/calib/{f}.txt', f'{KITTI_PATH_CALIBS}/validation/calib/.'])
