import cv2 as cv
import numpy as np
from pathlib import Path
import logging
import warnings

log = logging.getLogger(__name__)
rng = np.random.default_rng(seed=20231205)


def calibrate_camera(
    path_images,
    pattern_size=(9, 6),
    square_size_mm=25,  # https://stackoverflow.com/a/46052474
    flags_detector=cv.CALIB_CB_FAST_CHECK
    + cv.CALIB_CB_ADAPTIVE_THRESH
    + cv.CALIB_CB_NORMALIZE_IMAGE,
    display=False,
    save=False,
    save_path=Path(".", "camera-params.npz"),
):
    # Defining the world coordinates for 3D points
    _points_3d = np.zeros((1, pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    _points_3d[0, :, :2] = (
        np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
        * square_size_mm
    )

    points_3d = []  # 3d point in real world space
    points_2d = []  # 2d points in image plane

    for p in path_images:
        img_bgr = cv.imread(str(p))
        img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        success, corners = cv.findChessboardCorners(
            img_gray, pattern_size, flags=flags_detector
        )

        # If desired number of corner are detected,
        # we refine the pixel coordinates and display
        # them on the images of checker board

        if success:
            # refine pixel coordinates for given 2d points
            corners = cv.cornerSubPix(
                img_gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            points_3d.append(_points_3d)
            points_2d.append(corners)

        if display:
            img_bgr = cv.drawChessboardCorners(
                img_bgr,
                patternSize=pattern_size,
                corners=corners,
                patternWasFound=success,
            )
            cv.imshow("Detected corners", img_bgr)
            cv.waitKey(0)

    # Performing camera calibration by passing the value of known 3D points
    # and corresponding pixel coordinates of the detected corners
    camera_params = cv.calibrateCamera(
        points_3d, points_2d, img_gray.shape[::-1], None, None
    )  # ret, mtx, dist, rvecs, tvecs

    camera_params = dict(
        retval=camera_params[0],
        cameraMatrix=camera_params[1],
        distCoeffs=camera_params[2],
        rvecs=camera_params[3],
        tvecs=camera_params[4],
    )

    log.info("Reprojection error")
    mean_error = 0
    points_2d_hat = []

    for i in range(len(points_3d)):
        points_2d_hat, _ = cv.projectPoints(
            points_3d[i],
            camera_params["rvecs"][i],
            camera_params["tvecs"][i],
            camera_params["cameraMatrix"],
            camera_params["distCoeffs"],
        )

        error = cv.norm(points_2d[i], points_2d_hat, cv.NORM_L2) / len(points_2d_hat)
        mean_error += error

    log.info(f"total error: {mean_error / len(points_2d_hat)}")

    if save:
        log.info(f"Saving camera parameters at {save_path}")
        np.savez(save_path, **camera_params)

    return camera_params


def undistort_image(img, camera_matrix, distorsion, alpha=1):
    # load image
    h, w = img.shape[:2]

    # Finetune camera matrix on the new image
    # getOptimalNewCameraMatrix is used to use different resolutions
    # from the same camera with the same calibration
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(
        camera_matrix, distorsion, imageSize=(w, h), alpha=alpha, newImgSize=(w, h)
    )

    # undistort
    dst = cv.undistort(img, camera_matrix, distorsion, None, new_camera_matrix)

    # crop the image
    x, y, w, h = roi
    return dst[y : y + h, x : x + w]


def detect_features(img, n_features=500):
    detector = cv.SIFT.create(n_features)
    keypoints, descriptors = detector.detectAndCompute(img, mask=None)
    return (keypoints, descriptors)


def match_features(
    descriptors0,
    descriptors1,
    # algorithm=0,
    # n_trees=5,
    threshold=0.5,
    n_features=None,
):
    # index_params = dict(algorithm=algorithm, trees=n_trees)
    # search_params = dict()
    # matcher = cv.FlannBasedMatcher(index_params, search_params)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(descriptors0, descriptors1, k=2)

    # From: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
    # DMatch.distance - Distance between descriptors. The lower, the better it is.
    # DMatch.trainIdx - Index of the descriptor in train descriptors
    # DMatch.queryIdx - Index of the descriptor in query descriptors
    # DMatch.imgIdx - Index of the train image.

    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    if not len(good_matches):
        warnings.warn("No good features match found", RuntimeWarning)

    if n_features is not None:
        idxs = rng.choice(len(good_matches), min(n_features, len(good_matches)))
        good_matches = (np.asarray(good_matches)[idxs]).tolist()
    return good_matches


def camera_pose(keypoints0, keypoints1, matches, camera_matrix):
    # From: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
    # DMatch.distance - Distance between descriptors. The lower, the better it is.
    # DMatch.trainIdx - Index of the descriptor in train descriptors
    # DMatch.queryIdx - Index of the descriptor in query descriptors
    # DMatch.imgIdx - Index of the train image.
    # Also see https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python
    points0 = [keypoints0[i.queryIdx].pt for i in matches]
    points1 = [keypoints1[i.trainIdx].pt for i in matches]

    points0 = np.asarray(points0)
    points1 = np.asarray(points1)
    # points0 = np.asarray(points0).round()
    # points1 = np.asarray(points1).round()

    E, mask_inliers = cv.findEssentialMat(
        points1=points0,
        points2=points1,
        cameraMatrix=camera_matrix,
        method=cv.RANSAC,
        prob=0.99,  # default 0.999
        threshold=1.0,  # default 1.0
    )

    # https://stackoverflow.com/questions/77522308/understanding-cv2-recoverposes-coordinate-frame-transformations
    inliers0 = np.asarray(points0)
    inliers1 = np.asarray(points1)

    _, R, t, _ = cv.recoverPose(
        E=E,
        points1=inliers0,
        points2=inliers1,
        cameraMatrix=camera_matrix,
        mask=mask_inliers,
    )

    log.debug(f"R: \n{R.round(2)}")
    log.debug(f"t: \n{t.squeeze(),round(2)}")

    return (R, t)


def video_properties(video_path):
    video = cv.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(f"Error opening video {video_path}")

    frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    fps = video.get(cv.CAP_PROP_FPS)
    fourcc = video.get(cv.CAP_PROP_FOURCC)

    return {
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "fourcc": fourcc,
    }


def read_video(video_path):
    video = cv.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(f"Error opening video {video_path}")

    while True:
        success, frame_bgr = video.read()
        if not success:
            break

        yield frame_bgr

    video.release()


def write_video(video_path, frames, fps=30):
    frames_first = frames[0]
    frame_size = frames_first.shape[1], frames_first.shape[0]

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    video = cv.VideoWriter(
        str(video_path), fourcc=fourcc, fps=fps, frameSize=frame_size
    )

    for frame in frames:
        video.write(frame)

    video.release()


def draw_text(
    img,
    text,
    pos=(0, 0),
    font=cv.FONT_HERSHEY_DUPLEX,
    font_scale=1,
    font_thickness=1,
    text_color=(255, 255, 255),
    text_color_bg=(0, 0, 0),
    text_margin=10,
):
    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(
        img,
        (pos[0] - text_margin, pos[1] - text_margin),
        (x + text_w + text_margin, y + text_h + text_margin),
        text_color_bg,
        -1,
    )
    cv.putText(img, text, (x, y + text_h), font, font_scale, text_color, font_thickness)

    return img
