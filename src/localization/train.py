"""
Notes:
    In the paper, the pixel-wise softmax was used.
    But, I used the IOU because the datasets I used are
    not labeled for segmentations

Original Paper:
    https://arxiv.org/abs/1505.04597
"""
import os
import time

import pandas as pd
import tensorflow as tf

HEIGHT = 640
WIDTH = 1088


def image_augmentation(image, mask):
    """
    Random flip (left <--> right)
    Random brightness
    Random hue
    """
    concat_image = tf.concat([image, mask], axis=-1)

    flipped = tf.image.random_flip_left_right(concat_image)

    image = flipped[:, :, :-1]
    mask = flipped[:, :, -1:]

    image = tf.image.random_brightness(image, 0.7)
    image = tf.image.random_hue(image, 0.3)

    return image, mask


def get_image_mask(queue, augmentation=True):
    """Returns `image` and `mask`

    Input pipeline:
        Queue -> CSV -> FileRead -> Decode JPEG

    (1) Queue contains a CSV filename
    (2) Text Reader opens the CSV
        CSV file contains two columns
        ["path/to/image.jpg", "path/to/mask.jpg"]
    (3) File Reader opens both files
    (4) Decode JPEG to tensors

    Notes:
        height, width = HEIGHT, WIDTH

    Returns
        image (3-D Tensor): (HEIGHT, WIDTH, 3)
        mask (3-D Tensor): (HEIGHT, WIDTH, 1)
    """
    text_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_content = text_reader.read(queue)

    image_path, mask_path = tf.decode_csv(
        csv_content, record_defaults=[[""], [""]])

    image_file = tf.read_file(image_path)
    mask_file = tf.read_file(mask_path)

    image = tf.image.decode_jpeg(image_file, channels=3)
    image.set_shape([HEIGHT, WIDTH, 3])
    image = tf.cast(image, tf.float32)

    mask = tf.image.decode_jpeg(mask_file, channels=1)
    mask.set_shape([HEIGHT, WIDTH, 1])
    mask = tf.cast(mask, tf.float32)
    mask = mask / (tf.reduce_max(mask) + 1e-7)

    if augmentation:
        image, mask = image_augmentation(image, mask)

    return image, mask


def conv_conv_pool(input_,
                   n_filters,
                   training,
                   flags,
                   name,
                   pool=True,
                   activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}

    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions

    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=None,
                padding='same',
                # kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upconv_concat(inputA, input_B, n_filter, flags, name):
    """Upsample `inputA` and concat with `input_B`

    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation

    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D(inputA, n_filter, flags, name)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))


def upconv_2D(tensor, n_filter, flags, name):
    """Up Convolution `tensor` by 2 times

    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations

    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.layers.conv2d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        # kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        name="upsample_{}".format(name))


def make_unet(X, training, flags=None):
    """Build a U-Net architecture

    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers

    Returns:
        output (4-D Tensor): (N, H, W, C)
            Same shape as the `input` tensor

    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    net = X / 127.5 - 1
    conv1, pool1 = conv_conv_pool(net, [8, 8], training, flags, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, flags, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, flags, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, flags, name=4)
    conv5 = conv_conv_pool(
        pool4, [128, 128], training, flags, name=5, pool=False)

    up6 = upconv_concat(conv5, conv4, 64, flags, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, flags, name=6, pool=False)

    up7 = upconv_concat(conv6, conv3, 32, flags, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, flags, name=7, pool=False)

    up8 = upconv_concat(conv7, conv2, 16, flags, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, flags, name=8, pool=False)

    up9 = upconv_concat(conv8, conv1, 8, flags, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, flags, name=9, pool=False)

    return tf.layers.conv2d(
        conv9,
        1, (1, 1),
        name='final',
        activation=tf.nn.sigmoid,
        padding='same')


def IOU_(y_pred, y_true):
    """Returns a (approx) IOU score

    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7

    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)

    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(
        pred_flat, axis=1) + tf.reduce_sum(
        true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)


def make_train_op(y_pred, y_true):
    """Returns a training operation

    Loss function = - IOU(y_pred, y_true)

    IOU is

        (the area of intersection)
        --------------------------
        (the area of two boxes)

    Args:
        y_pred (4-D Tensor): (N, H, W, 1)
        y_true (4-D Tensor): (N, H, W, 1)

    Returns:
        train_op: minimize operation
    """
    loss = -IOU_(y_pred, y_true)

    global_step = tf.train.get_or_create_global_step()

    optim = tf.train.AdamOptimizer(learning_rate=0.0006)
    return optim.minimize(loss, global_step=global_step)


def read_flags():
    """Returns flags"""

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--epochs", default=10, type=int, help="Number of epochs")

    parser.add_argument("--batch-size", default=12, type=int, help="Batch size")

    parser.add_argument(
        "--logdir", default="logdir", help="Tensorboard log directory")

    parser.add_argument(
        "--reg", type=float, default=0.0001, help="L2 Regularizer Term")

    parser.add_argument(
        "--ckdir", default="models", help="Checkpoint directory")

    flags = parser.parse_args()
    return flags


def main(flags):
    train = pd.read_csv("./train.csv")
    n_train = train.shape[0]

    test = pd.read_csv("./test.csv")
    n_test = test.shape[0]

    current_time = time.strftime("%m/%d/%H/%M/%S")
    train_logdir = os.path.join(flags.logdir, "train", current_time)
    test_logdir = os.path.join(flags.logdir, "test", current_time)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 3], name="X")
    y = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 1], name="y")
    mode = tf.placeholder(tf.bool, name="mode")

    pred = make_unet(X, mode, flags)

    tf.add_to_collection("inputs", X)
    tf.add_to_collection("inputs", mode)
    tf.add_to_collection("outputs", pred)

    tf.summary.histogram("Predicted Mask", pred)
    tf.summary.image("Predicted Mask", pred)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = make_train_op(pred, y)

    IOU_op = IOU_(pred, y)
    IOU_op = tf.Print(IOU_op, [IOU_op])
    tf.summary.scalar("IOU", IOU_op)

    train_csv = tf.train.string_input_producer(['train.csv'])
    test_csv = tf.train.string_input_producer(['test.csv'])
    train_image, train_mask = get_image_mask(train_csv)
    test_image, test_mask = get_image_mask(test_csv, augmentation=False)

    X_batch_op, y_batch_op = tf.train.shuffle_batch(
        [train_image, train_mask],
        batch_size=flags.batch_size,
        capacity=flags.batch_size * 5,
        min_after_dequeue=flags.batch_size * 2,
        allow_smaller_final_batch=True)

    X_test_op, y_test_op = tf.train.batch(
        [test_image, test_mask],
        batch_size=flags.batch_size,
        capacity=flags.batch_size * 2,
        allow_smaller_final_batch=True)

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_summary_writer = tf.summary.FileWriter(test_logdir)

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()
        if os.path.exists(flags.ckdir) and tf.train.checkpoint_exists(
                flags.ckdir):
            latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
            saver.restore(sess, latest_check_point)

        else:
            try:
                os.rmdir(flags.ckdir)
            except FileNotFoundError:
                pass
            os.mkdir(flags.ckdir)

        try:
            global_step = tf.train.get_global_step(sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for epoch in range(flags.epochs):

                for step in range(0, n_train, flags.batch_size):
                    X_batch, y_batch = sess.run([X_batch_op, y_batch_op])

                    _, step_iou, step_summary, global_step_value = sess.run(
                        [train_op, IOU_op, summary_op, global_step],
                        feed_dict={X: X_batch,
                                   y: y_batch,
                                   mode: True})

                    train_summary_writer.add_summary(step_summary, global_step_value)

                total_iou = 0
                for step in range(0, n_test, flags.batch_size):
                    X_test, y_test = sess.run([X_test_op, y_test_op])
                    step_iou, step_summary = sess.run(
                        [IOU_op, summary_op],
                        feed_dict={X: X_test,
                                   y: y_test,
                                   mode: False})

                    total_iou += step_iou * X_test.shape[0]

                    test_summary_writer.add_summary(step_summary, global_step_value)

            saver.save(sess, "{}/model.ckpt".format(flags.ckdir))

        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model.ckpt".format(flags.ckdir))


if __name__ == '__main__':
    flags = read_flags()
    main(flags)
