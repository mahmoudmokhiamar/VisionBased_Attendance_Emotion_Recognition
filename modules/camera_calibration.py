import cv2
import numpy as np
import os

def calibrate_camera(calib_images_dir="data/calibration", chessboard_size=(9, 6)):
    """
    Calibrate the camera using chessboard images in the specified directory.
    Returns the camera matrix and distortion coefficients.
    """
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    for fname in os.listdir(calib_images_dir):
        path = os.path.join(calib_images_dir, fname)
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise ValueError("Calibration failed: No corners found.")

    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist
