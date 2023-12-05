import logging
from pathlib import Path
import cv2 as cv
from scipy.spatial.transform import Rotation
import funcs

log_format = (
    "%(asctime)s" "[%(levelname)s]" "%(filename)s %(funcName)s:%(lineno)d - %(message)s"
)

logging.basicConfig(format=log_format, level=logging.DEBUG)
log = logging.getLogger(__name__)

log.info("Start camera calibration process...")
path_calibration_images = list(Path("data", "calibration").glob("*.jpg"))
data_processed_path = Path("data", "processed")
data_processed_path.mkdir(exist_ok=True, parents=True)
camera_params_path = Path(data_processed_path, "camera-params.npz")

camera = funcs.calibrate_camera(
    path_images=path_calibration_images,
    display=False,
    save=True,
    save_path=camera_params_path,
)

log.info("Loading video...")
vid_path = Path("data", "external", "still.mp4")
vid_props = funcs.video_properties(vid_path)

frames = funcs.read_video(vid_path)

all_frames = list(frames)
frame_t0 = all_frames[0]

frame_t0_gray = cv.cvtColor(frame_t0, cv.COLOR_BGR2GRAY)
frame_t0_gray = funcs.undistort_image(
    frame_t0_gray, camera["cameraMatrix"], camera["distCoeffs"]
)

log.info("Estimate camera pose...")
for iframe, frame_t1 in enumerate(all_frames):
    log.info(f"Processing frame {iframe}")

    frame_t1_gray = cv.cvtColor(frame_t1, cv.COLOR_BGR2GRAY)

    frame_t1_gray = funcs.undistort_image(
        frame_t1_gray, camera["cameraMatrix"], camera["distCoeffs"]
    )

    k0, d0 = funcs.detect_features(frame_t0_gray)
    k1, d1 = funcs.detect_features(frame_t1_gray)
    matches = funcs.match_features(d0, d1, n_features=25)

    debug_frame = frame_t1_gray.copy()
    if len(matches):
        R, t = funcs.camera_pose(
            keypoints0=k0,
            keypoints1=k1,
            matches=matches,
            camera_matrix=camera["cameraMatrix"],
        )

        # R_angle = Rotation.from_matrix(R).inv().as_euler("xyz", degrees=True)
        R_angle = Rotation.from_matrix(R).as_quat()
        log.debug(f"Quat: \n{R_angle.round(2)}")

        debug_frame = funcs.draw_text(
            img=debug_frame,
            # text=f"frame: {iframe:05d} R(x, y, z, w)={R_angle.round(1).squeeze()} t = {t.round(1).squeeze()}",
            text=f"frame: {iframe:05d} R={R.round(2).tolist()}",
        )

        debug_frame = cv.drawMatches(
            frame_t0_gray,
            k0,
            debug_frame,
            k1,
            matches,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        cv.imshow("Feature match", debug_frame)
        cv.waitKey(0)

    frame_t0_gray = frame_t1_gray.copy()

cv.destroyAllWindows()
