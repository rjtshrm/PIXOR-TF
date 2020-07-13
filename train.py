from tensorflow.keras import optimizers, backend
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datagen import  DataGen
from tf_model import pixor_modified
from loss import custom_loss, class_loss, reg_loss
import tensorflow as tf


def lr_scheduler(initial_lr):

    def scheduler(epoch):
        if epoch > 45:
            epoch = 44
        epoch = epoch // 15
        lr = initial_lr * tf.math.pow(0.1, epoch).numpy()
        print(lr)
        return lr

    return scheduler


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    model = pixor_modified(input_shape=(800, 700, 36), num_block=[3, 6, 6, 3], use_bn=True, use_height=False)
    PREFIX_PATH = '/home/rash8327/Downloads/kitti'
    KITTI_PATH_VELODYNE = '{0}/data_object_velodyne/training/velodyne/'.format(PREFIX_PATH)
    KITTI_PATH_LABELS = '{0}/data_object_label_2/training/label_2/'.format(PREFIX_PATH)
    KITTI_PATH_CALIBS = '{0}/data_object_calib/training/calib/'.format(PREFIX_PATH)

    VAL_KITTI_PATH_VELODYNE = '{0}/data_object_velodyne/validation/velodyne/'.format(PREFIX_PATH)
    VAL_KITTI_PATH_LABELS = '{0}/data_object_label_2/validation/label_2/'.format(PREFIX_PATH)
    VAL_KITTI_PATH_CALIBS = '{0}/data_object_calib/validation/calib/'.format(PREFIX_PATH)

    log_dir = 'logs/pixor_modified'
    checkpoint_dir = 'checkpoint'

    datagenerator = DataGen(KITTI_PATH_VELODYNE, KITTI_PATH_LABELS, KITTI_PATH_CALIBS, use_cache=True, batch_size=3, type='train', augmentation=True, use_height=False, norm=False)
    valdatagenerator = DataGen(VAL_KITTI_PATH_VELODYNE, VAL_KITTI_PATH_LABELS, VAL_KITTI_PATH_CALIBS, use_cache=True, batch_size=3, type='val', augmentation=False, use_height=False, norm=False)

    weight_decay = 0.0001
    for layer in model.layers:
        for attr in ['kernel_regularizer', 'bias_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, tf.keras.regularizers.l2(weight_decay))

    model = tf.keras.models.model_from_json(model.to_json())

    lr = 0.001
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=custom_loss(class_weight=1.0, reg_weight=1.0),
                  metrics=[class_loss, reg_loss],
                  run_eagerly=True
                  )

    modecheckpoint_cb = ModelCheckpoint(f'{checkpoint_dir}/cp-{{epoch:04d}}.ckpt', monitor='loss', save_weights_only=True)

    tensorboard_cb = TensorBoard(log_dir=log_dir)

    scheduler_cb = tf.keras.callbacks.LearningRateScheduler(lr_scheduler(lr))

    initial_epoch = 0

    if checkpoint_dir:
        print('checking checkpoints')
        cps = tf.train.latest_checkpoint('checkpoint')
        if cps:
            print(f'Checkpoint found : {cps}')
            initial_epoch = int(cps.split('/')[-1].split('.')[0].split('-')[-1])
            model.load_weights(cps)


    model.fit_generator(
        generator=datagenerator,
        validation_data=valdatagenerator,
        initial_epoch=initial_epoch,
        epochs=80,
        callbacks=[modecheckpoint_cb, tensorboard_cb, scheduler_cb],
        use_multiprocessing=True,
        workers=5
    )

    print('Training Finished')
