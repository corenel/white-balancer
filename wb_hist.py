import argparse
import cv2
import math
import numpy as np
import sys


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix


def calibrate_white_balance(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0
    channels = cv2.split(img)
    calib_params = []

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)
        n_cols = flat.shape[0]

        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

        calib_params.append((low_val, high_val))

    return calib_params


def apply_white_balance(img, calib_params):
    assert img.shape[2] == 3
    channels = cv2.split(img)
    out_channels = []
    for idx, channel in enumerate(channels):
        assert len(channel.shape) == 2
        low_val = calib_params[idx][0]
        high_val = calib_params[idx][1]

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255,
                                   cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='White balancer by LAB color space')
    parser.add_argument('calib_image_path',
                        type=str,
                        help='path to calibration image')
    parser.add_argument('--test', '-t'
                        type=str,
                        help='path to test image')
    args = parser.parse_args()
    # Calculate parameters
    calib_img = cv2.imread(args.calib_image_path)
    calib_params = calibrate_white_balance(calib_img, 1)
    # Apply and show
    calib_img_out = apply_white_balance(calib_img, calib_params)
    cv2.imshow("calib_img (before)", calib_img)
    cv2.imshow("calib_img (after)", calib_img_out)
    if args.test is not None:
        test_img = cv2.imread(sys.argv[2])
        test_img_out = apply_white_balance(test_img, calib_params)
        cv2.imshow("test_img (before)", test_img)
        cv2.imshow("test_img (after)", test_img_out)
    cv2.waitKey(0)
