import sys

import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage.measurements import label

from ThreadVideoCapture import ThreadVideoCapture

CLASSIFIER_IMG_SIZE = 90
DRAW_GRID = False

TEST_IMAGE = "testimg/test.ppm"
TEST_VIDEO = "testvideo/test1.mp4"
MIN_SCORE_THRESH = 0.23
MIN_BOX_SIZE = 15
GRID_CACHE_SIZE = 5

sys.path.append("/home/cn1h/PycharmProjects/models/research")
sys.path.append("/home/cn1h/PycharmProjects/models/research/object_detection")
from object_detection_api import label_map_util
from object_detection_api import visualization_utils as vis_util

PATH_TO_LABELS = 'GTSDB_label_map.pbtxt'
NUM_CLASSES = 43

image_WH = (1088, 640)


# check if the new box inside any old box
def already_drawn_bbox(bbox, top, left, bottom, right):
    for t, l, b, r in bbox:
        if l <= left <= r:
            if l <= right <= r:
                if t <= top <= b:
                    if t <= bottom <= b:
                        return True

    return False


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


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def find_boxes(image, sess_unet, pred, X, mode):
    """
    :return:
    boxes, each is [top, left, bottom, right]
    img_signs, each is image with shape (CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE)
    """
    softmax_value = sess_unet.run(pred, feed_dict={X: np.array([image]), mode: False})
    softmax_value = np.squeeze(softmax_value)
    softmax_value = softmax_value > 0.9999

    # in labeled_heatmap all connected 1 points (found area) have the same label
    labeled_heatmap, n_labels = label(softmax_value)

    boxes = []
    img_signs = []

    for i in range(n_labels):
        mask_i = labeled_heatmap == (i + 1)

        # no zero index (two arrays)
        nonzero = np.nonzero(mask_i)

        nonzero_row = nonzero[0]
        nonzero_col = nonzero[1]

        left = min(nonzero_col)
        top = min(nonzero_row)
        right = max(nonzero_col)
        bottom = max(nonzero_row)

        if not already_drawn_bbox(boxes, top, left, bottom, right):
            cut_out_area = image[top:bottom + 1, left:right + 1, :]
            # skip too small boxes
            if cut_out_area.shape[0] < MIN_BOX_SIZE or cut_out_area.shape[1] < MIN_BOX_SIZE:
                continue
            cut_out_area = resize_image_with_pad(cut_out_area, CLASSIFIER_IMG_SIZE)
            img_signs.append(cut_out_area)
            boxes.append([top, left, bottom, right])

    return np.array(boxes), np.array(img_signs)


def run_inference_for_single_image(image, sess_unet, pred, X, mode, sess_classifier, input, is_training, softmax):
    """
    :param image, it must be resized to correct size
    :return: a dict with follow keys:

    detection_classes: class of each box
    detection_boxes: shape: (c, 4), 4 is for: ymin, xmin, ymax, xmax
    detection_scores: confidence about each classification

    None if nothing found

    """

    output_dict = {}

    boxes, img_signs = find_boxes(image, sess_unet, pred, X, mode)

    if img_signs.size == 0:
        return None

    softmax_value = sess_classifier.run(softmax, feed_dict={input: img_signs, is_training: False})

    output_dict['detection_scores'] = np.max(softmax_value, axis=1)
    output_dict['detection_classes'] = np.argmax(softmax_value, axis=1) + 1
    output_dict['detection_boxes'] = boxes

    return output_dict


# divide screen into 10 x 10 grids, cache the box class values inside each grid
n_grid_each_axis = 10
width_each_grid, height_each_grid = image_WH[0] / n_grid_each_axis, image_WH[1] / n_grid_each_axis
# deque: https://stackoverflow.com/a/7913731/1943272
grid_cache = [[{"classes": GRID_CACHE_SIZE * [999],  # no class: 999
                "scores": GRID_CACHE_SIZE * [0]}
               for _ in range(n_grid_each_axis)] for _ in range(n_grid_each_axis)]


def update_grid_cache(new_grid_value):
    for i in range(n_grid_each_axis):
        for j in range(n_grid_each_axis):
            cache = grid_cache[i][j]
            new_value = new_grid_value[i][j]
            cache["classes"].pop(0)
            cache["classes"].append(new_value["classes"])
            cache["scores"].pop(0)
            cache["scores"].append(new_value["scores"])


def run_inference_for_multi_image(image, sess_unet, pred, X, mode, sess_classifier, input, is_training, softmax):
    """
    Different than run_inference_for_single_image, it will take average of recent frames to decide class and class score

    :param image, it must be resized to correct size
    :return: a dict with follow keys:

    detection_classes: class of each box
    detection_boxes: shape: (c, 4), 4 is for: ymin, xmin, ymax, xmax
    detection_scores: confidence about each classification

    None if nothing found

    """

    output_dict = {}

    boxes, img_signs = find_boxes(image, sess_unet, pred, X, mode)

    new_grid_value = [[{"classes": 999, "scores": 0} for _ in range(n_grid_each_axis)] for _ in range(n_grid_each_axis)]

    if img_signs.size == 0:
        update_grid_cache(new_grid_value)
        return None

    softmax_value = sess_classifier.run(softmax, feed_dict={input: img_signs, is_training: False})

    scores = np.max(softmax_value, axis=1)
    classes = np.argmax(softmax_value, axis=1) + 1

    for i in range(len(boxes)):
        score = scores[i]
        classe = classes[i]
        [top, left, bottom, right] = boxes[i]
        center_h = top + (bottom - top) // 2
        center_w = left + (right - left) // 2

        # // is ok, it always returns [0,len-1] because center h/w can never be max h or w
        h_index = int(center_h // height_each_grid)
        w_index = int(center_w // width_each_grid)

        # skip some areas
        if h_index > 7:  # skip last 2 rows
            scores[i] = 0
            classes[i] = 999
            continue
        if h_index < 3:  # skip first 2 rows and in middle
            if 1 < w_index < 8:
                scores[i] = 0
                classes[i] = 999
                continue

        # update new grid value
        new_grid_value[h_index][w_index] = {"classes": classe, "scores": score}

        # update result with averages (not include the current value)
        cache = grid_cache[h_index][w_index]
        scores[i] = np.mean(cache["scores"])
        # calculate mode: https://stackoverflow.com/a/6252400/1943272
        classes[i] = np.bincount(cache["classes"]).argmax()

    update_grid_cache(new_grid_value)

    # filter out invalid class after averaging
    valid_index = classes != 999

    output_dict['detection_scores'] = scores[valid_index]
    output_dict['detection_classes'] = classes[valid_index]
    output_dict['detection_boxes'] = boxes[valid_index]

    return output_dict


def test_with_image():
    graph_unet = tf.Graph()
    graph_classifier = tf.Graph()
    with tf.Session(graph=graph_unet) as sess_unet:
        with tf.Session(graph=graph_classifier) as sess_classifier:
            with graph_unet.as_default():
                saver_unet = tf.train.import_meta_graph('models/' + "models_27_8" + '/model.ckpt.meta')
                saver_unet.restore(sess_unet, "models/" + "models_27_8/" + "model.ckpt")
                X, mode = tf.get_collection("inputs")
                pred = tf.get_collection("outputs")[0]

            with graph_classifier.as_default():
                saver_classifier = tf.train.import_meta_graph('outs/my_train_classifier/trained.meta')
                saver_classifier.restore(sess_classifier, "outs/my_train_classifier/trained")
                input = graph_classifier.get_tensor_by_name("placeholder/input:0")
                is_training = graph_classifier.get_tensor_by_name("placeholder/is_training:0")
                softmax = graph_classifier.get_tensor_by_name("softmax:0")

            image = cv2.imread(TEST_IMAGE)
            image = resize_to_nn_support_size(image)

            # Actual detection.
            output_dict = run_inference_for_single_image(
                image, sess_unet, pred, X, mode, sess_classifier, input, is_training, softmax)
            print_detection(output_dict)

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=False,
                line_thickness=4,
                skip_scores=True,
                min_score_thresh=MIN_SCORE_THRESH
            )

            cv2.imshow("test_with_image", image)
            cv2.waitKey(0)
            cv2.imwrite("testimg/test_out.jpg", image)
            cv2.destroyAllWindows()


# https://stackoverflow.com/a/49807766/1943272
def draw_grid(img, line_color=(0, 255, 0), thickness=1):
    x = width_each_grid
    y = height_each_grid
    while x < img.shape[1]:
        cv2.line(img, (int(x), 0), (int(x), img.shape[0]), color=line_color, lineType=cv2.LINE_AA, thickness=thickness)
        x += width_each_grid

    while y < img.shape[0]:
        cv2.line(img, (0, int(y)), (img.shape[1], int(y)), color=line_color, lineType=cv2.LINE_AA, thickness=thickness)
        y += height_each_grid


def test_with_video():
    # the size must be same as input video
    video_out = cv2.VideoWriter('output_thread.avi',
                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                25,  # output video fps
                                image_WH)

    video_capture = ThreadVideoCapture(capture_interval_in_sec=0.035, device_index=(TEST_VIDEO))
    video_capture.start()

    graph_unet = tf.Graph()
    graph_classifier = tf.Graph()
    with tf.Session(graph=graph_unet) as sess_unet:
        with tf.Session(graph=graph_classifier) as sess_classifier:
            with graph_unet.as_default():
                saver_unet = tf.train.import_meta_graph('models/models_27_8/model.ckpt.meta')
                saver_unet.restore(sess_unet, "models/models_27_8/model.ckpt")
                X, mode = tf.get_collection("inputs")
                pred = tf.get_collection("outputs")[0]

            with graph_classifier.as_default():
                saver_classifier = tf.train.import_meta_graph('outs/my_train_classifier/trained.meta')
                saver_classifier.restore(sess_classifier, "outs/my_train_classifier/trained")
                input = graph_classifier.get_tensor_by_name("placeholder/input:0")
                is_training = graph_classifier.get_tensor_by_name("placeholder/is_training:0")
                softmax = graph_classifier.get_tensor_by_name("softmax:0")

            while True:
                frame = video_capture.get_last_frame()
                if frame is None:
                    continue

                frame = resize_to_nn_support_size(frame)

                # Actual detection.
                output_dict = run_inference_for_multi_image(
                    frame, sess_unet, pred, X, mode, sess_classifier, input, is_training, softmax)

                if DRAW_GRID:
                    draw_grid(frame)

                # Visualization of the results of a detection.
                if output_dict is not None:
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        frame,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=False,
                        line_thickness=3,
                        min_score_thresh=MIN_SCORE_THRESH
                    )

                cv2.imshow("test_with_video", frame)
                # save frame to video
                video_out.write(frame)
                # this short 1ms pause is important, otherwise see nothing in windows
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    video_capture.stop()
    cv2.destroyAllWindows()


def resize_to_nn_support_size(frame):
    H, W, C = frame.shape
    if (W, H) != image_WH:
        return cv2.resize(frame, image_WH)
    else:
        return frame


def print_detection(output_dict):
    cs = output_dict['detection_classes']
    for i in range(len(cs)):
        score = output_dict['detection_scores'][i]
        if score >= MIN_SCORE_THRESH:
            print(category_index.get(cs[i])['name'], score)


# test_with_image()
test_with_video()
