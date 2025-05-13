import os
import cv2
import glob
import numpy as np
from typing import Tuple

def calibrate_camera(calib_images_dir="data/calibration", 
                    chessboard_size=(9,6), 
                    cache_file="camera_params.npz"):
    """
    Basic camera calibration using chessboard pattern.
    Returns (camera_matrix, dist_coeffs) or raises ValueError.
    """
    # Try to load cached calibration
    if os.path.isfile(cache_file):
        with np.load(cache_file) as data:
            return data["camera_matrix"], data["dist_coeffs"]

    # Prepare object points
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

    # Find chessboard corners
    objpoints, imgpoints = [], []
    images = glob.glob(os.path.join(calib_images_dir, "*.jpg")) + \
             glob.glob(os.path.join(calib_images_dir, "*.png"))

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Check we have enough images
    if len(objpoints) < 5:
        raise ValueError(f"Need at least 5 calibration images, found {len(objpoints)}")

    # Calibrate camera
    ret, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    if not ret:
        raise ValueError("Camera calibration failed")

    # Save cache
    np.savez(cache_file, camera_matrix=mtx, dist_coeffs=dist)
    return mtx, dist

def undistort_image(img, mtx, dist):
    """Simple undistortion wrapper"""
    return cv2.undistort(img, mtx, dist)