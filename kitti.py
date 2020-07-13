import copy
import numpy as np
import matplotlib.pyplot as plt
import utils

DEBUG = False
EPISLON = 1e-5
class KITTI(object):
    kitti_image_wid = 1382
    kitti_image_hgt = 512

    FRONT_RANGE = (0, 70)
    SIDE_RANGE = (-40, 40)
    VERTICAL_RANGE = (-2.5, 1.0)
    RES = 0.1
    DIST_THRESHOLD = 10 # in meters
    VEH_LEN_MAX = 5.24 #7.0
    VEH_LEN_MIN = 2.19 #2.0
    VEH_WID_MAX = 2.04 #3.0
    VEH_WID_MIN = 1.14 #1.3

    class_types = [
        'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
        'Cyclist', 'Tram', 'Misc', 'DontCare'
    ]

    colors = {
        'Car': 'r',
        'Tram': 'r',
        'Cyclist': 'g',
        'Van': 'c',
        'Truck': 'm',
        'Pedestrian': 'y',
        'Sitter': 'k',
        'Misc': 'w'
    }

    @staticmethod
    def read_pcd_bin(file):
        pcd = np.fromfile(file, dtype=np.float32, count=-1).reshape([-1,4])
        return pcd

    @staticmethod
    def read_label(file, lidar_2_cam, R0):
        if file == None:
            return []

        label = np.loadtxt(file, delimiter=' ', dtype='str')
        if len(label.shape) == 1:
            label = label.reshape(1, label.shape[0])
        tracklet = []

        for l in range(label.shape[0]):
            if label[l, 0] != 'Car': # detection only for cars as of now
                continue
            if float(label[l, 1]) >= 0.5: # if truncations is greater than 50%
                continue
            tklt = {
                'class': label[l, 0],
                'truncated': float(label[l, 1]),
                'occluded': int(label[l, 2]),
                'alpha': float(label[l, 3]),
                'bbox': {
                    'left': float(label[l, 4]),
                    'top': float(label[l, 5]),
                    'right': float(label[l, 6]),
                    'bottom': float(label[l, 7])
                },
                'dimensions': {
                    'height': float(label[l, 8]),
                    'width': float(label[l, 9]),
                    'length': float(label[l, 10]),
                }
            }

            roty = float(label[l, 14]) # w.r.t camera coords
            tklt['rotation_z'] =  KITTI.ry_to_rz(roty)

            cam = np.ones([4, 1])
            cam[0], cam[1], cam[2] = float(label[l, 11]), float(label[l, 12]), float(label[l, 13]) # x, y, z
            xyz = KITTI.project_cam2velo(cam, R0, lidar_2_cam)

            tklt['location'] = {
                'x': xyz[0, 0],
                'y': xyz[0, 1],
                'z': xyz[0, 2],
            }


            tracklet.append(tklt)
        return tracklet

    @staticmethod
    def read_calib(file):
        sensor_calibs = open(file).readlines()

        # Index 5: Tr_velo_to_cam (lidar to cam coordinates
        lidar_2_cam = np.array(sensor_calibs[5].split(' '))[1:]
        lidar_2_cam = np.concatenate([lidar_2_cam.reshape(3, 4).astype('float32'),
                                        np.array([0., 0., 0., 1.]).reshape(1, 4)], axis=0)

        # Index 4: Rotation matrix w.r.t cam view
        rot_2_cam = np.eye(4)
        rot_2_cam[:3, :3] = np.array(sensor_calibs[4].split(' '))[1:].reshape(3, 3).astype('float32')

        # Index 2: Projection matrix CAM 2
        p_cam = np.array(sensor_calibs[2].split(' '))[1:].reshape(3, 4).astype('float32')

        return lidar_2_cam, rot_2_cam, p_cam

    @staticmethod
    def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2 * np.pi + angle
        return angle

    @staticmethod
    def project_cam2velo(cam, r0, lidar_2_cam):
        t_inv = np.linalg.inv(lidar_2_cam)
        xyz = np.dot(t_inv, np.linalg.inv(r0) @ cam)[:3]
        return xyz.reshape(1, 3)

    @staticmethod
    def get_absolute_box(tracklet):
        l, w, h = tracklet['dimensions']['length'], tracklet['dimensions']['width'], tracklet['dimensions']['height']

        # orientation
        yaw = tracklet['rotation_z']

        # center
        translation = np.array([tracklet['location']['x'], tracklet['location']['y'], tracklet['location']['z']])

        rotMat = utils.rotation_z(yaw)

        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])

        bbx_cords = (rotMat @ trackletBox) + np.tile(translation, (8, 1)).T

        return bbx_cords

    @staticmethod
    def draw_box(tracklet, color='black', axis=None):
        # axis == None, plot bounding bpxes for 3d (xyz)

        vertices = KITTI.get_absolute_box(tracklet)

        if axis == 'xy':
            vertices = vertices[0:2, :]

        connections = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
            [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
            [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
        ]

        for connection in connections:
            plt.plot(*vertices[:, connection], c=color, lw=0.5)

    @staticmethod
    def get_pcd_idx_in_cam_fov(pcd, lidar_2_cam, rot_2_cam, p_cam):
        """
        :param pcd: point cloud N x 3
        :param lidar_2_cam: Matrix (4 x 4; R|T matrix)
        :param rot_2_cam: Camera view rotation
        :param p_cam: projection matrix
        :return: pcd with cam reference and in fov
        """
        # make homeogeneous coords
        pcd = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=-1).T

        # pcd with reference to camera
        pcd = lidar_2_cam @ pcd

        # pcd rotation in camera frame
        pcd = rot_2_cam @ pcd

        # pcd in camera projection
        pcd_2_cam = p_cam @ pcd
        pcd_2_cam = pcd_2_cam / pcd_2_cam[2, :]
        #print(pcd_2_cam.shape)

        # point in the field of view of camera
        idx = (pcd_2_cam[0, :] >= 0) & (pcd_2_cam[0, :] <= KITTI.kitti_image_wid) & \
              (pcd_2_cam[1, :] >= 0) & (pcd_2_cam[1, :] <= KITTI.kitti_image_hgt)


        return idx

    @staticmethod
    def plot_bev(velodyne_file, calib_file, label_file, plot_label=True, points=0.1):
        pcd = KITTI.read_pcd_bin(velodyne_file)
        lidar_2_cam, rot_2_cam, p_cam = KITTI.read_calib(calib_file)

        pcd = pcd[pcd[:, 0] >= 0, :] # drop -x axis back of vehicle
        idx = KITTI.get_pcd_idx_in_cam_fov(pcd[:, 0:3], lidar_2_cam, rot_2_cam, p_cam)
        pcd = pcd[idx, :]

        #np.savetxt('pcd.xyz', pcd[:, 0:3], delimiter=' ', newline='\n')

        # Drop pcd z and intensity and add 1 to xy
        xy = pcd[:, 0:2]
        point_size = 0.01 * (1. / points)
        points_step = int(1. / points)
        velo_range = range(0, xy.shape[0], points_step)
        pcd = xy[velo_range, :]

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)

        ax.scatter(xy[:, 0], xy[:, 1], s=point_size, cmap='grey')

        if plot_label:
            tracklets = KITTI.read_label(label_file, lidar_2_cam=lidar_2_cam, R0=rot_2_cam)

            for tklt in tracklets:
                KITTI.draw_box(tklt, color=KITTI.colors[tklt['class']], axis='xy')

        plt.savefig('temp.png')
        plt.show()

    @staticmethod
    def get_processed_data(velodyne_file, calib_file, label_file, side_range=SIDE_RANGE,
                           front_range=FRONT_RANGE, vertical_range=VERTICAL_RANGE, res=RES, z_res=RES,
                           augmentation=True, use_height=False, raw_lidar=False, normalize_gt=False):
        """
            Preprocess KITTI Data: Transforms Lidar Point Cloud to discretized 2d grid at discretized heights
            :param velodyne_file: lidar file path
            :param calib_file: calibration file path
            :param label_file: tracklets labels
            :return BEV's of Lidar and Labels grid (class + localization)
        """
        pcd = KITTI.read_pcd_bin(velodyne_file)
        lidar_2_cam, rot_2_cam, p_cam = KITTI.read_calib(calib_file)
        tracklets = KITTI.read_label(label_file, lidar_2_cam=lidar_2_cam, R0=rot_2_cam)

        pcd = pcd[pcd[:, 0] >= 0, :]  # drop -x axis back of vehicle

        idx = KITTI.get_pcd_idx_in_cam_fov(pcd[:, 0:3], lidar_2_cam, rot_2_cam, p_cam)
        pcd = pcd[idx, :]

        if augmentation:
            if np.random.random() > 0.5:
                # rand rotation
                rotz = utils.rotation_z(np.random.uniform(-18, 18) * np.pi / 180.)  # (-π/10 to π/10)
                pcd[:, 0:3] = (rotz @ pcd[:, 0:3].T).T
                for tklt in tracklets:
                    xyz = np.ones([3, 1])
                    xyz[0], xyz[1], xyz[2] = tklt['location']['x'], tklt['location']['y'], tklt['location']['z']
                    xyz = (rotz @ xyz).reshape(3)
                    tklt['location']['x'], tklt['location']['y'], tklt['location']['z'] = xyz[0], xyz[1], xyz[2]
            if np.random.random() > 0.5:
                # rand scaling
                scale_factor = np.random.uniform(0.95, 1.05)
                pcd[:, 0:3] = pcd[:, 0:3] * scale_factor
                for tklt in tracklets:
                    for xyz in tklt['location']:
                        tklt['location'][xyz] = tklt['location'][xyz] * scale_factor
                    for lwh in tklt['dimensions']:
                        tklt['dimensions'][lwh] = tklt['dimensions'][lwh] * scale_factor

        x_size = int((front_range[1] - front_range[0]) / res)
        y_size = int((side_range[1] - side_range[0]) / res)
        z_size = int((vertical_range[1] - vertical_range[0]) / z_res) + 1  # + intensity

        velodyne_output_shape = (y_size, x_size, z_size)

        velodyne_data = KITTI.point_cloud_top_as_channels(pcd, velodyne_output_shape, side_range=side_range,
                                    front_range=front_range, vertical_range=vertical_range, res=res, z_res=res, use_height=use_height)


        if label_file == None:
            # Only for testing
            class_grid, loc_grid =  None, None
        else:
            class_grid, loc_grid = KITTI.get_label_as_distance_grid(tracklets, output_shape=(velodyne_output_shape[0], velodyne_output_shape[1]),
                                                                                  side_range=side_range, front_range=front_range, vertical_range=vertical_range,
                                                                                  res=res, z_res=res, use_height=use_height, normalize_gt=normalize_gt)

        if raw_lidar:
            # only for inference
            return velodyne_data, class_grid, loc_grid, pcd
        else:
            return velodyne_data, class_grid, loc_grid, None

    @staticmethod
    def get_unrotated_box(tracklet):
        l, w, h = tracklet['dimensions']['length'], tracklet['dimensions']['width'], tracklet['dimensions']['height']
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])

        return trackletBox

    @staticmethod
    def normalize(d, min, max):
        return (d - min) / (max - min)

    @staticmethod
    def de_normalize(d, min, max):
        return d * (max - min) + min

    @staticmethod
    def get_label_as_distance_grid(tracklets, output_shape, side_range,
                                    front_range, vertical_range, res, z_res, use_height, normalize_gt):
        downsample_factor = 4
        output_shape = np.array(output_shape) // downsample_factor
        class_grid = np.zeros(output_shape, dtype=np.float32)
        # loc_grid contains {cos(θ), sin(θ), dx, dy, z, log(w), log(l), log(h)}
        if use_height:
            op_channels = 8
        else:
            op_channels = 6

        loc_grid = np.zeros((output_shape[0], output_shape[1], op_channels), dtype=np.float32)

        x = np.linspace(front_range[0], front_range[1], output_shape[1])
        y = -1 * np.linspace(side_range[0], side_range[1], output_shape[0])
        xx_orig, yy_orig = np.meshgrid(x, y, indexing='xy')

        for tklt in tracklets:
            if tklt['location']['x'] > front_range[1]:
                continue
            if np.abs(tklt['location']['y']) > side_range[1]:
                continue

            xx = np.copy(xx_orig)
            yy = np.copy(yy_orig)

            x_centre, y_centre, z_centre = tklt['location']['x'], tklt['location']['y'], tklt['location']['z']
            l, w, h = tklt['dimensions']['length'], tklt['dimensions']['width'], tklt['dimensions']['height']
            x_delta = x_centre - xx
            y_delta = y_centre - yy

            # Handle Rotation: w.r.t to world coordinates bbx
            rotz = tklt['rotation_z']
            xx_r = x_delta * np.cos(-rotz) - y_delta * np.sin(-rotz)
            yy_r = x_delta * np.sin(-rotz) + y_delta * np.cos(-rotz)

            # get all point inside rotated bbv_corners
            x_mask = np.logical_and((xx_r >= -l/2.), (xx_r <= l/2.))
            y_mask = np.logical_and((yy_r >= -w/2.), (yy_r <= w/2.))
            xy_mask = np.logical_and(x_mask, y_mask)

            # normalize delta_x, delta_y (0 to 1)
            #x_delta = (x_delta - (-KITTI.VEH_LEN_MAX)*0.5) / ((KITTI.VEH_LEN_MAX - (-KITTI.VEH_LEN_MAX))*0.5)
            #y_delta = (y_delta - (-KITTI.VEH_WID_MAX)*0.5) / ((KITTI.VEH_WID_MAX - (-KITTI.VEH_WID_MAX))*0.5)
            #cost = (np.cos(rotz) - (-1.)) / (1. - (-1.))
            #sint = (np.sin(rotz) - (-1.)) / (1. - (-1.))
            #l = (np.log(l) - np.log(KITTI.VEH_LEN_MIN)) / (np.log(KITTI.VEH_LEN_MAX) - np.log(KITTI.VEH_LEN_MIN))
            #w = (np.log(w) - np.log(KITTI.VEH_WID_MIN)) / (np.log(KITTI.VEH_WID_MAX) - np.log(KITTI.VEH_WID_MIN))
            cost = np.cos(rotz)
            sint = np.sin(rotz)
            l = np.log(l)
            w = np.log(w)

            if normalize_gt:
                x_delta = KITTI.normalize(x_delta, -KITTI.VEH_LEN_MAX*0.5, KITTI.VEH_LEN_MAX*0.5)
                y_delta = KITTI.normalize(y_delta, -KITTI.VEH_WID_MAX*0.5, KITTI.VEH_WID_MAX*0.5)
                cost = KITTI.normalize(cost, -1., 1.)
                sint = KITTI.normalize(sint, -1., 1.)
                l = KITTI.normalize(l, np.log(KITTI.VEH_LEN_MIN), np.log(KITTI.VEH_LEN_MAX))
                w = KITTI.normalize(w, np.log(KITTI.VEH_WID_MIN), np.log(KITTI.VEH_WID_MAX))

            if use_height:
                # TO Do
                h = 0
            else:
                h = 0

            class_grid[xy_mask] = 1   # probability of object existence


            if use_height:
                gt_loc_grid = [cost, sint, x_delta[xy_mask],
                                             y_delta[xy_mask], z_centre, l, w, h]
            else:
                gt_loc_grid = [cost, sint, x_delta[xy_mask],
                               y_delta[xy_mask], l, w]

            for i in range(len(gt_loc_grid)):
                loc_grid[..., i][xy_mask] = gt_loc_grid[i]

        return class_grid,  loc_grid


    @staticmethod
    def decode_labels(loc_grid, output_shape=(200, 175), side_range=(-40, 40), front_range=(0, 70), vertical_range=(-2.5, 1.0), res=0.1, z_res=0.1, use_height=False, de_normalize=False):
        # TO DO: Height
        cost = loc_grid[..., 0]
        sint = loc_grid[..., 1]
        x_delta = loc_grid[..., 2]
        y_delta = loc_grid[..., 3]
        idx = 4

        if use_height:
            z = loc_grid[..., idx]
            h = loc_grid[..., idx + 2]
            idx += 1

        l = loc_grid[..., idx]
        w = loc_grid[..., idx + 1]


        if de_normalize:
            cost = KITTI.de_normalize(cost, -1. , 1.)
            sint = KITTI.de_normalize(sint, -1., 1.)
            x_delta = KITTI.de_normalize(x_delta, -KITTI.VEH_LEN_MAX * 0.5, KITTI.VEH_LEN_MAX * 0.5)
            y_delta = KITTI.de_normalize(y_delta, -KITTI.VEH_WID_MAX * 0.5, KITTI.VEH_WID_MAX * 0.5)
            l = KITTI.de_normalize(l, np.log(KITTI.VEH_LEN_MIN), np.log(KITTI.VEH_LEN_MAX))
            w = KITTI.de_normalize(w, np.log(KITTI.VEH_WID_MIN), np.log(KITTI.VEH_WID_MAX))
            if use_height:
                # TO DO
                pass

        l = np.exp(l)
        w = np.exp(w)
        if use_height:
            # TO DO:
            #h = np.exp(h)
            pass

        theta = np.arctan2(sint, cost)

        x = np.linspace(front_range[0], front_range[1], output_shape[1])
        y = -1 * np.linspace(side_range[0], side_range[1], output_shape[0])

        xx, yy = np.meshgrid(x, y, indexing='xy')

        # centers
        x_center = xx + x_delta
        y_center = yy + y_delta

        return utils.extract_2d_vertices_from_anchors(theta, x_center, y_center, l, w)



    @staticmethod
    def point_cloud_top_as_channels(pcd, output_shape, side_range,
                                    front_range, vertical_range, res, z_res, use_height):
        """
        :param side_range: y-range (side of vehicle)
        :param front_range: x-range (front of vehicle)
        :param vertical_range: z-range (vertical of vehicle)
        :param res: x , y res
        :param z_res: z res
        :return: output feature of point cloud
        """
        x_min, x_max = front_range
        y_min, y_max = side_range
        z_min, z_max = vertical_range

        # EXTRACT THE POINTS FOR EACH AXIS
        x_points = pcd[:, 0]
        y_points = pcd[:, 1]
        z_points = pcd[:, 2]
        reflectance = pcd[:, 3]

        velo_processed = np.zeros(output_shape, dtype=np.float32)
        intensity_map_count = np.zeros((output_shape[0], output_shape[1]), dtype=np.int)

        # remove all points which are not in range
        # forward filter
        f_filt = np.logical_and(
            (x_points > x_min), (x_points < x_max))
        # side filter
        s_filt = np.logical_and(
            (y_points > y_min), (y_points < y_max))

        filt = np.logical_and(f_filt, s_filt)

        for i, hgt in enumerate(np.arange(z_min, z_max, z_res)):
            z_filt = np.logical_and((z_points >= hgt),
                                    (z_points < hgt + z_res))
            z_filt = np.logical_and(filt, z_filt)

            indices = np.argwhere(z_filt).flatten()

            # get all valid points
            xi_points = x_points[indices]
            yi_points = y_points[indices]
            zi_points = z_points[indices]
            ref_i = reflectance[indices]

            # Convert x, y to pixel position values
            x_img = (xi_points / res).astype(np.int32)
            y_img = (-yi_points / res).astype(np.int32) # y is -y axis lidar to camera

            # zero centered
            x_img -= int(np.floor(x_min / res))
            y_img -= int(np.floor(y_min / res))

            if use_height:
                # TO DO: HANDLE
                pixel_value = hgt + (z_res * 0.5) # center of one discrete channel, could be mean value or some other approach
                pixel_value = pixel_value / z_res
            else:
                pixel_value = 2
            velo_processed[y_img, x_img, i] = pixel_value
            velo_processed[y_img, x_img, -1] += ref_i
            intensity_map_count[y_img, x_img] += 1
        intensity_map_count[intensity_map_count == 0] = 1
        velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1],  intensity_map_count)

        if DEBUG:
            for i in range(output_shape[-1]):
                v = velo_processed[:, :, i]
                idx = np.where(v > 0)
                plt.scatter(idx[1], idx[0], s=0.1)

            # set axes range
            plt.xlim(0, output_shape[1])
            plt.ylim(output_shape[0], 0)
            plt.show()


        return velo_processed


if __name__ == '__main__':
    DEBUG = True
    KITTI_PATH_VELODYNE = '/mnt/raid_data/srr7rng/KITTI/data_object_velodyne/training/velodyne/'
    KITTI_PATH_LABELS = '/mnt/raid_data/srr7rng/KITTI/data_object_label_2/training/label_2/'
    KITTI_PATH_CALIBS = '/mnt/raid_data/srr7rng/KITTI/data_object_calib/training/calib/'
    k = KITTI()
    k.plot_bev(velodyne_file=KITTI_PATH_VELODYNE+'004330.bin', calib_file=KITTI_PATH_CALIBS+'004330.txt', label_file=KITTI_PATH_LABELS+'004330.txt')
    vp = k.get_processed_data(velodyne_file=KITTI_PATH_VELODYNE+'004330.bin', calib_file=KITTI_PATH_CALIBS+'004330.txt', label_file=KITTI_PATH_LABELS+'004330.txt')
