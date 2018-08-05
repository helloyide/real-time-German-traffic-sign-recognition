import os

import cv2
import numpy as np
import numpy.random as random
import tensorflow as tf
from keras.utils import to_categorical

from classification import squeezenet_v1_1

n_test = 4000
n_class = 43
image_width_height = 90


# https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
def resize_image_with_pad(img, desired_size):
    old_size = img.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=color)
    return img


def read_image_and_resize(image_path, resize_to):
    image = cv2.imread(image_path)
    image = resize_image_with_pad(image, resize_to)
    return image


def parse_annotation_file(image_path):
    """
    :return: list of annotation, which contains:
    "file_name": the image full filename
    "class_label": label index, starts from 1
    """
    annotations = []

    for i in range(43):
        label_index = str(i + 1)
        with tf.gfile.GFile(os.path.join(image_path, label_index, 'gt.csv'), 'r') as fid:
            lines = fid.readlines()
        first_line = True
        for line in lines:
            if first_line:
                first_line = False
                continue
            file_name, _, _, xmin, ymin, xmax, ymax, _ = line.split(";")
            annotation = {"file_name": os.path.join(image_path, label_index, file_name.replace(".ppm", ".jpg")),
                          "class_label": label_index}
            annotations.append(annotation)

    return annotations


annotations = parse_annotation_file('datasets/GTSRB/train/images')
random.seed(0)
random.shuffle(annotations)

annotations_test = annotations[:n_test]
annotations_train = annotations[n_test:]

train_x = []
train_y = []
for annotation in annotations_train:
    file_name = annotation["file_name"]
    label = int(annotation["class_label"])
    train_x.append(read_image_and_resize(file_name, image_width_height))
    train_y.append(label - 1)

test_x = []
test_y = []
for annotation in annotations_test:
    file_name = annotation["file_name"]
    label = int(annotation["class_label"])
    test_x.append(read_image_and_resize(file_name, image_width_height))
    test_y.append(label - 1)

# turn to one-hot
train_y = to_categorical(train_y, n_class)
test_y = to_categorical(test_y, n_class)
train_x = np.array(train_x)
test_x = np.array(test_x)

m = train_x.shape[0]
lr = 0.0003
n_epoch = 20
n_batch_size = 128
reg_lambda = 1e-5
keep_prob = 0.8

with tf.variable_scope("placeholder"):
    x = tf.placeholder(tf.float32, (None, image_width_height, image_width_height, 3), name="input")
    y = tf.placeholder(tf.float32, (None, n_class))
    is_training = tf.placeholder(tf.bool, (), name="is_training")

logits = squeezenet_v1_1.inference(x, keep_prob, is_training, n_class, reg_lambda)
logits = tf.identity(logits, 'output')
softmax = tf.nn.softmax(logits, name="softmax")

with tf.variable_scope("loss"):
    loss_base = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss_total = loss_base + tf.reduce_sum(reg_losses)

with tf.variable_scope("evaluation"):
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(logits, axis=-1), tf.argmax(y, axis=-1)),
                tf.float32), name="accuracy")

with tf.variable_scope("train"):
    global_step = tf.get_variable("global_step", shape=(), dtype=tf.int32, trainable=False)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_total, global_step=global_step)

with tf.variable_scope("summary"):
    summary_loss_total = tf.summary.scalar("loss_total", loss_total)
    summary_accuracy_test = tf.summary.scalar("accuracy_test", accuracy)
    summary_accuracy_train = tf.summary.scalar("accuracy_train", accuracy)

with tf.variable_scope("utils"):
    saver = tf.train.Saver()


# TODO
def standardization(x):
    return x


normalized_test_x = standardization(test_x)

logdir = "outs/my_train_classifier/"
with tf.Session() as sess, tf.summary.FileWriter(
        logdir,
        graph=tf.get_default_graph()) as f:
    sess.run(tf.global_variables_initializer())

    # similar logic as mnist's next_batch()
    epoch = 0
    index_in_epoch = 0
    while epoch < n_epoch:
        for _ in range(m // n_batch_size + 1):
            start = index_in_epoch
            if start + n_batch_size > m:
                epoch += 1
                n_rest_data = m - start
                train_x_batch_rest = train_x[start:m]
                train_y_batch_rest = train_y[start:m]
                # Shuffle train data
                perm = np.arange(m)
                np.random.shuffle(perm)
                train_x = train_x[perm]
                train_y = train_y[perm]
                # Start next epoch
                start = 0
                index_in_epoch = n_batch_size - n_rest_data
                end = index_in_epoch
                train_x_batch_new = train_x[start:end]
                train_y_batch_new = train_y[start:end]
                # concatenate
                train_x_batch = np.concatenate((train_x_batch_rest, train_x_batch_new), axis=0)
                train_y_batch = np.concatenate((train_y_batch_rest, train_y_batch_new), axis=0)
            else:
                index_in_epoch += n_batch_size
                end = index_in_epoch
                train_x_batch = train_x[start:end]
                train_y_batch = train_y[start:end]

            _, global_step_value, loss_total_value, summary_loss_total_value = \
                sess.run([train_op, global_step, loss_total, summary_loss_total],
                         feed_dict={x: standardization(train_x_batch),
                                    y: train_y_batch,
                                    is_training: True})

            if global_step_value % 100 == 0:
                accuracy_train_value, summary_accuracy_train_value = \
                    sess.run([accuracy, summary_accuracy_train],
                             feed_dict={x: standardization(train_x_batch),
                                        y: train_y_batch,
                                        is_training: False})
                accuracy_test_value, summary_accuracy_test_value = \
                    sess.run([accuracy, summary_accuracy_test],
                             feed_dict={x: normalized_test_x,
                                        y: test_y,
                                        is_training: False})

                print(global_step_value, epoch, loss_total_value, accuracy_train_value, accuracy_test_value)

                f.add_summary(summary_loss_total_value, global_step=global_step_value)
                f.add_summary(summary_accuracy_train_value, global_step=global_step_value)
                f.add_summary(summary_accuracy_test_value, global_step=global_step_value)

    saver.save(sess, logdir + "/trained")
