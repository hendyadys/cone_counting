import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
# plt.switch_backend('agg')


def load_mask(file_name):
    mask_name = '{}_mask.png'.format(file_name.split('.png')[0])
    mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    return mask


def load_img(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    return img


def plot_boxes_cv(img, coords):
    for bbox in coords:
        (x1, y1, x2, y2) = (bbox[0], bbox[1], bbox[2], bbox[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imshow('img', img)
    return


def compute_centers(coords):
    # coord_centers_x = []
    # coord_centers_y = []
    centers = []
    for coord in coords:
        (real_x1, real_y1, real_x2, real_y2) = coord
        # coord_centers_x.append((real_x1 + real_x2) / 2.0)
        # coord_centers_y.append((real_y1 + real_y2) / 2.0)
        centers.append(( (real_x1 + real_x2) / 2.0, (real_y1 + real_y2) / 2.0 ))
    return centers
    # return coord_centers_x, coord_centers_y


def compute_cone_coords(x, y, num_rows=32, num_cols=32):
    cone_size = 7
    x1 = max(x-(cone_size//2), 0)             # lower bound
    x2 = min(x+(cone_size//2), num_rows-1)   # upper bound
    y1 = max(y-(cone_size//2), 0)             # lower bound
    y2= min(y+(cone_size//2), num_cols-1)    # upper bound
    return x1, y1, x2, y2


def plot_centers_np(img, centers, flipAxis=False, savepath=None):
    plt.figure()
    plt.imshow(img)

    coord_centers_x = []
    coord_centers_y = []
    for coord in centers:
        coord_centers_x.append(coord[0])
        coord_centers_y.append(coord[1])

    if flipAxis:
        plt.scatter(x=coord_centers_y, y=coord_centers_x, c='red', s=10)  # imshow flips coords
        plt.title('img and centers flipped')
    else:
        plt.scatter(x=coord_centers_x, y=coord_centers_y, c='blue', s=10)
        plt.title('img and centers')

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()
    return


def plot_boxes_np(img, coords, flipAxis=False, savepath=None):
    fig1, ax1 = plt.subplots(1)
    ax1.imshow(img)
    for coord in coords:
        (real_x1, real_y1, real_x2, real_y2) = coord
        if flipAxis:
            ax1.add_patch(
                patches.Rectangle((real_y1, real_x1), real_y2 - real_y1, real_x2 - real_x1, fill=False, color='red'))
        else:
            ax1.add_patch(
                patches.Rectangle((real_x1, real_y1), real_x2 - real_x1, real_y2 - real_y1, fill=False, color='blue'))

    plt.show()

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()
    return


def visualize_cone_data(file_name, coords):
    img = load_img(file_name)
    img2 = img.copy()

    # parsed coords from generated data
    plot_boxes_cv(img, coords)
    centers = compute_centers(coords)
    plot_centers_np(img2, centers)
    plot_centers_np(img2, centers, flipAxis=True)
    plot_boxes_np(img2, coords)
    plot_boxes_np(img2, coords, flipAxis=True)

    # # check mask for coords
    # mask = load_mask(file_name)
    # mask_coords = np.transpose(np.nonzero(mask))    # find cone centers in mask file
    # box_coords = []
    # for coord in mask_coords:
    #     (x1, y1, x2, y2) = compute_cone_coords(coord[0], coord[1])
    #     box_coords.append((x1, y1, x2, y2))
    #
    # plot_boxes_cv(img, box_coords)
    # centers2 = compute_centers(box_coords)
    # plot_centers_np(img2, centers2)
    # plot_centers_np(img2, centers2, flipAxis=True)
    # plot_boxes_np(img2, box_coords)
    # plot_boxes_np(img2, box_coords, flipAxis=True)