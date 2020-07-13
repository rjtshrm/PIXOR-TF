import math
import os
import glob
from tqdm import tqdm
import logging
from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
from kitti import KITTI

logging.getLogger().setLevel(logging.INFO)

class DataGen(Sequence):
    '''
        KITTI DataLoader
    '''
    
    def __init__(self, kitti_path_velodyne, kitti_path_label, kitti_path_calib, type, batch_size=5, use_cache=False, shuffle=True, augmentation=False, use_height=False, raw_lidar=False, norm=False):
        self.kitti_path_velodyne = kitti_path_velodyne
        self.kitti_path_label = kitti_path_label
        self.kitti_path_calib = kitti_path_calib
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.type = type
        self.augmentation = augmentation
        self.use_height = use_height
        self.norm = norm

        self.velodyne = pd.DataFrame(columns=['frame_id', 'file_path'])
        self.label = pd.DataFrame(columns=['frame_id', 'file_path'])
        self.calib = pd.DataFrame(columns=['frame_id', 'file_path'])
        self.raw_lidar = raw_lidar
        self.use_cache = use_cache

        if not self.use_cache:
            self.read_files()
        else:
            try:
                self.velodyne = pd.read_pickle(f'cache/{self.type}/velodyne.pkl')
                self.label = pd.read_pickle(f'cache/{self.type}/label.pkl') if self.kitti_path_label else None
                self.calib = pd.read_pickle(f'cache/{self.type}/calib.pkl')
            except Exception as e:
                logging.info('Cache can\'t found. Reading files')
                # create cache directory
                os.makedirs(f'cache/{self.type}', exist_ok=True)
                self.read_files()

        self.idx = np.arange(len(self.velodyne))



    def on_epoch_end(self):
        if self.shuffle: # called after every epoch
           np.random.shuffle(self.idx)

    def read_files(self):
        self.read_velodyne()
        self.read_labels()
        self.read_calibs()

    def read_velodyne(self):
        files = glob.glob(os.path.join(self.kitti_path_velodyne, '*.bin'))
        logging.info('Reading all velodyne files')
        for f_path in tqdm(files):
            frame_id = f_path.split('/')[-1][:-4]
            self.velodyne = self.velodyne.append({'frame_id': frame_id, 'file_path': f_path}, ignore_index=True)

        if self.use_cache:
            pd.to_pickle(self.velodyne, f'cache/{self.type}/velodyne.pkl')


    def read_labels(self):
        if self.kitti_path_label == None:
            return
        files = glob.glob(os.path.join(self.kitti_path_label, '*.txt'))
        logging.info('Reading all label files')
        for f_path in tqdm(files):
            frame_id = f_path.split('/')[-1][:-4]
            self.label = self.label.append({'frame_id': frame_id, 'file_path': f_path}, ignore_index=True)

        if self.use_cache:
            pd.to_pickle(self.label, f'cache/{self.type}/label.pkl')

    def read_calibs(self):
        files = glob.glob(os.path.join(self.kitti_path_calib, '*.txt'))
        logging.info('Reading all calib files')
        for f_path in tqdm(files):
            frame_id = f_path.split('/')[-1][:-4]
            self.calib = self.calib.append({'frame_id': frame_id, 'file_path': f_path}, ignore_index=True)

        if self.use_cache:
            pd.to_pickle(self.calib, f'cache/{self.type}/calib.pkl')

    def __len__(self):
        return math.ceil(self.velodyne.shape[0] / self.batch_size)

    def __getitem__(self, index):
        # velodyne
        v_batch_data = np.empty((self.batch_size, 800, 700, 36), dtype=np.float32)
        # class + tracklet
        if self.use_height:
            op_channels = 1 + 8
        else:
            op_channels = 1 + 6
        c_t_batch_data = np.empty((self.batch_size, 200, 175, op_channels), dtype=np.float32) if self.kitti_path_label else None
        pcd_batch = []

        #'Generate one batch of data'
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        velodyne_files = self.velodyne.iloc[self.idx[start_idx:end_idx]]

        btch_idx = 0
        for index, row in velodyne_files.iterrows():
            frame_id = row.frame_id
            velodyne_file = row.file_path

            if self.kitti_path_label:
                label_file = self.label.loc[self.label['frame_id'] == frame_id].file_path.item()
            else:
                label_file = None

            calib_file = self.calib.loc[self.calib['frame_id'] == frame_id].file_path.item()

            # velodyne top view 36 channels
            # c types of object present
            # tracklets geometry
            v, c, t, pcd = KITTI.get_processed_data(
                velodyne_file, calib_file, label_file, augmentation=self.augmentation, use_height=self.use_height, raw_lidar=self.raw_lidar, normalize_gt=self.norm
            )

            v_batch_data[btch_idx, ...] = v

            if self.raw_lidar: pcd_batch.append(pcd)

            if self.kitti_path_label:
                c = np.expand_dims(c, axis=-1)

                c_t_batch_data[btch_idx, ...] = np.concatenate((c, t), axis=-1)

            btch_idx += 1

        if self.raw_lidar:
            return v_batch_data, c_t_batch_data, pcd_batch
        else:
            return v_batch_data, c_t_batch_data



if __name__ == '__main__':
    TRAIN_KITTI_PATH_VELODYNE = '/home/rajat/Downloads/kitti/data_object_velodyne/training/velodyne/'
    TRAIN_KITTI_PATH_LABELS = '/home/rajat/Downloads/kitti/data_object_label_2/training/label_2/'
    TRAIN_KITTI_PATH_CALIBS = '/home/rajat/Downloads/kitti/data_object_calib/training/calib/'

    VAL_KITTI_PATH_VELODYNE = '/home/rajat/Downloads/kitti/data_object_velodyne/validation/velodyne/'
    VAL_KITTI_PATH_LABELS = '/home/rajat/Downloads/kitti/data_object_label_2/validation/label_2/'
    VAL_KITTI_PATH_CALIBS = '/home/rajat/Downloads/kitti/data_object_calib/validation/calib/'
    td = DataGen(TRAIN_KITTI_PATH_VELODYNE, TRAIN_KITTI_PATH_LABELS, TRAIN_KITTI_PATH_CALIBS, use_cache=True, type='train', batch_size=1)
    vd = DataGen(VAL_KITTI_PATH_VELODYNE, VAL_KITTI_PATH_LABELS, VAL_KITTI_PATH_CALIBS, use_cache=True, type='eval', batch_size=1)

    print(len(td), len(vd))
    t = []
    e = []
    for i in range (len(td)):
        print(i)
        v , c_t = td.__getitem__(i)
        t.append(v[..., 35].max())
        t.append(v[..., 35].min())
    for i in range (len(vd)):
        print(i)
        v , c_t = vd.__getitem__(i)
        e.append(v[..., 35].max())
        e.append(v[..., 35].min())
