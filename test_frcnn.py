from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
"Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.",
                  default='resnet50')
# parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.",
#                   default='vgg')

(options, args) = parser.parse_args()

if not options.test_path:  # if filename is not given
    parser.error('Error: path to test data must be specified. Pass --path to command line')

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

print('C.network=%s' % C.network)
if C.network == 'resnet50':
    import keras_frcnn.resnet as nn
elif C.network == 'vgg':
    import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path


def plot_cone_preds(file_name, img, coords, flipAxis=False, is_all=False):
    from matplotlib import pyplot as plt
    from matplotlib import patches
    plt.switch_backend('agg')

    fig1, ax1 = plt.subplots(1)
    ax1.imshow(img)
    for coord in coords:
        (real_x1, real_y1, real_x2, real_y2) = coord
        if flipAxis: # imshow flips coords
            ax1.add_patch(
                patches.Rectangle((real_y1, real_x1), real_y2 - real_y1, real_x2 - real_x1, fill=False, color='green'))
        else:
            ax1.add_patch(
                patches.Rectangle((real_x1, real_y1), real_x2 - real_x1, real_y2 - real_y1, fill=False, color='green'))
    img_name = '{}_{}.png'.format(file_name.split('.png')[0], 'all' if is_all else '')
    plt.savefig(img_name, bbox_inches='tight')

    # plot points on img
    plt.figure(2)
    coord_centers_x = []
    coord_centers_y = []
    for coord in coords:
        (real_x1, real_y1, real_x2, real_y2) = coord
        coord_centers_x.append((real_x1 + real_x2) / 2.0)
        coord_centers_y.append((real_y1 + real_y2) / 2.0)
    plt.imshow(img)
    if flipAxis:
        plt.scatter(x=coord_centers_y, y=coord_centers_x, c='blue', s=10)   # imshow flips coords
    else:
        plt.scatter(x=coord_centers_x, y=coord_centers_y, c='blue', s=10)
    img_name = '{}_{}centers.png'.format(file_name.split('.png')[0], 'all' if is_all else '')
    plt.savefig(img_name, bbox_inches='tight')
    # plt.show()
    return


def calc_euclidean_loss(true_coords, predicted_coords):
    num_coords_true = len(true_coords)
    num_preds = len(predicted_coords)
    euclidean_loss = np.zeros((num_coords_true, num_preds))
    for i, tc in enumerate(true_coords):
        for j,pc in enumerate(predicted_coords):
            temp = (tc[0]-pc[0])**2 + (tc[1]-pc[1])**2
            euclidean_loss[i,j] = np.sqrt(temp)

    return np.min(euclidean_loss, axis=1), num_coords_true, num_preds


def compute_metrics(file_name, img, coords, all_coords):
    mask_name = '{}_mask.png'.format(file_name.split('.png')[0])
    mask_data = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    mask_coords = np.transpose(np.nonzero(mask_data))

    euclidean_loss, num_coords_true, num_preds = calc_euclidean_loss(mask_coords, coords)
    euclidean_loss_all, num_coords_true_all, num_preds_all = calc_euclidean_loss(mask_coords, coords)
    return euclidean_loss, num_coords_true, num_preds, euclidean_loss_all, num_preds_all


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
    num_features = 1024
elif C.network == 'vgg':
    num_features = 512

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
print('img_input.shape={}; roi_input.shape={}; feature_map_input.shape={}; '
      'shared_layers.shape={}; num_anchors={}'.format(img_input.shape, roi_input.shape, feature_map_input.shape,
                                                      shared_layers.shape, num_anchors))
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True
debug_file = './debug.txt'
all_coords_file = '%s/all_coords.txt' % img_path
coords_file = '%s/coords.txt' % img_path
loss_file =  '%s/losses.txt' % img_path

for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    # if idx > 0: break
    if 'overview' in img_path and 'mask' in img_name:   # treat mask/truth separately
        continue

    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    print(img_name)
    st = time.time()
    filepath = os.path.join(img_path, img_name)

    img = cv2.imread(filepath)

    X, ratio = format_img(img, C)
    with open(debug_file, 'a') as fout:
        fout.write('ratio={} \n X.shape={}\n X={} \n'.format(ratio, X.shape, X))

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)
    with open(debug_file, 'a') as fout:
        fout.write('Y1.shape={}\n Y1={} \n'.format(Y1.shape, Y1))
        fout.write('Y2.shape={}\n Y2={} \n'.format(Y2.shape, Y2))
        fout.write('F.shape={}\n F={} \n'.format(F.shape, F))

    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
    with open(debug_file, 'a') as fout:
        fout.write('R.shape={}\n R={} \n'.format(R.shape, R))

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        with open(debug_file, 'a') as fout:
            fout.write('ROIs.shape={}\n ROIs={} \n'.format(ROIs.shape, ROIs))

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
        with open(debug_file, 'a') as fout:
            fout.write('P_cls.shape={}\n P_cls={}\n'.format(P_cls.shape, P_cls))
            fout.write('P_regr.shape={}\n P_regr={}\n'.format(P_regr.shape, P_regr))

        for ii in range(P_cls.shape[1]):
            print(np.max(P_cls[0, ii, :]), np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1))

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]
            print('first coords',x,y,w,h)

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                print('regressed coords', x, y, w, h)
            except:
                pass
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    with open(debug_file, 'a') as fout:
        fout.write('len(bboxes)={}\n bboxes={}\n'.format(len(bboxes), bboxes))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        supp_coords = []
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk, :]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
            supp_coords.append((real_x1, real_y1, real_x2, real_y2))

            if 'overview' not in img_path:
                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                              (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)

                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))
                all_dets.append((key, 100 * new_probs[jk]))

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (real_x1, real_y1 - 0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        # store all coords
        all_coords = []
        for i in range(len(bbox)):
            x1, y1, x2, y2 = bbox[i, :]
            all_coords.append(get_real_coordinates(ratio, x1, y1, x2, y2))

        if 'overview' in img_path:
            # visualize all_coords and thresholded coords
            plot_cone_preds(img_name, img, supp_coords, flipAxis=False, is_all=False)
            # plot_cone_preds(img_name, img, all_coords, flipAxis=False, is_all=True)
            euclidean_loss, num_coords_true, num_preds, euclidean_loss_all, num_preds_all = \
                compute_metrics('%s/%s' % (img_path, img_name), img, supp_coords, all_coords)

            with open(coords_file, 'a') as fout:
                fout.write('{}\t{}\n'.format(img_name, supp_coords))
            with open(all_coords_file, 'a') as fout:
                fout.write('{}\t{}\n'.format(img_name, all_coords))
            with open(loss_file, 'a') as fout:
                fout.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(img_name, num_coords_true, num_preds,
                                                             np.sum(euclidean_loss)/float(num_coords_true),
                                                             num_preds_all, np.sum(euclidean_loss_all)/float(num_coords_true)))

    print('Elapsed time = {}'.format(time.time() - st))
    print(all_dets)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.imwrite('./results_imgs/{}.png'.format(idx),img)


# command line arguments
#  "-p" "../overview/1-process-data/final/valid" "--network" "vgg"
# "-p" "../overview/1-process-data/testImages" "--network" "vgg"
# "-p" "../VOCdevkit/VOC2012/JPEGImages" "--network" "vgg"
# "-o" "simple" "-p" "../overview/1-process-data/cone_data_valid.txt" "--network" "vgg"
# "-p" "../VOCdevkit" "--network" "vgg"