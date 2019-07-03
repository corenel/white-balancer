import argparse
import cv2
import numpy as np

another_image_path = './3.jpg'
calib_image_path = './5.jpg'


def calibrate_white_balance(img):
    height, width, channels = img.shape
    upper_left = (int(width / 3), int(height / 3))
    bottom_right = (int(width * 2 / 3), int(height * 2 / 3))
    rect_img = img[upper_left[1]:bottom_right[1], upper_left[0]:
                   bottom_right[0]]
    result = cv2.cvtColor(rect_img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    return avg_a, avg_b


def apply_white_balance(img, avg_a, avg_b):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) *
                                         (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) *
                                         (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


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
    wb_a, wb_b = calibrate_white_balance(calib_img)
    # Apply and show
    calib_img_out = apply_white_balance(calib_img, wb_a, wb_b)
    cv2.imshow('calib image (before)', calib_img)
    cv2.imshow('calib image (after)', calib_img_out)
    if args.test is not None:
        test_img = cv2.imread(args.test)
        test_img_out = apply_white_balance(test_img, wb_a, wb_b)
        cv2.imshow('test image (before)', test_img)
        cv2.imshow('test image (after)', test_img_out)
    cv2.waitKey(0)
