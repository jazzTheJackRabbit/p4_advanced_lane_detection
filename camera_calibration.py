import pickle
import cv2
import numpy as np
import glob
from pipeline import show_image
from pdb import set_trace as b
from report_helper import *

show_image_bool = True


def read_image(image_path):
    return cv2.imread(image_path)


def calibrate(path='./camera_cal/calibration*.jpg'):
    image_paths = glob.glob(path)
    imagepoints = []
    objpoints = []

    nx = 9
    ny = 6

    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Convert to grayscale
    for image_path in image_paths:
        # print("processing {}".format(image_path))
        image = read_image(image_path)
        img = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        show_image(image, text="Gray Scale Image")
        
        # Find the chessboard corners
        corner_ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if corner_ret == True:
        # Draw and display the corners
            imagepoints.append(corners)
            objpoints.append(objp)
            cv2.drawChessboardCorners(image, (nx, ny), corners, corner_ret)

            show_image(img, text="Distored Image")
            

            # * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imagepoints, img.shape[1::-1], None, None)

            undistorted = cv2.undistort(img, mtx, dist, None, mtx)

            show_image(undistorted, text="Undistorted")
            

            src = np.float32([corners[0],corners[8],corners[35],corners[27]])

            w, h = img.shape[1::-1]

            top = 0.1; bottom = 0.55; left = 0.1; right = 0.9

            dst = np.float32([[w*left, h*top],[w*right, h*top],[w*right, h*bottom],[w*left, h*bottom]])

            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            M = cv2.getPerspectiveTransform(src, dst)

            # e) use cv2.warpPerspective() to warp your image to a top-down view
            warped = cv2.warpPerspective(undistorted, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
            show_image(warped, text="Perspective Warped image")
            

    calibration_values = {
        'imgpoints': imagepoints,
        'objpoints': objpoints,
        'mtx_distortion_correction': mtx,
        'distortion_coefficient': dist
    }
    pickle.dump(calibration_values, open("calibration_values.p", "wb" ) )