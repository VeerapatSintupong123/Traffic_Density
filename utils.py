import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

ROOT_TEXT = "./text"
ROOT_VIDEO = "./video"
ROOT_PICTURE = "./picture"


def video_to_frame(
    video_path: str, fps: int = 1, start_time: int = 0, end_time: int = None
):
    frames_dir = os.path.join(ROOT_PICTURE, "frames")

    if not os.path.exists(ROOT_PICTURE):
        os.makedirs(ROOT_PICTURE)
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    if os.path.exists(video_path):
        video = cv.VideoCapture(video_path)

        if not video.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        video_fps = video.get(cv.CAP_PROP_FPS)
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps) if end_time else total_frames

        if start_frame >= total_frames or start_frame >= end_frame:
            print("Error: Invalid start or end time.")
            video.release()
            return

        video.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = 0
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = video.read()
            if not ret:
                break

            if (current_frame - start_frame) % int(video_fps / fps) == 0:
                frame_filename = os.path.join(
                    frames_dir, f"frame_{frame_count:04d}.jpg"
                )
                cv.imwrite(frame_filename, frame)
                frame_count += 1

            current_frame += 1

        video.release()
        cv.destroyAllWindows()
        print(f"Extracted {frame_count} frames to {frames_dir}")
    else:
        print(f"Error: Video file {video_path} does not exist.")


def segment_road_and_car(img_path):
    image = cv.imread(img_path)

    # get height and width image
    (H, W) = image.shape[:2]

    # convert image to gray scale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # blur image before using edge detection algorithm (canny)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # edge detection algorithm
    canny = cv.Canny(blurred, 30, 150)

    # start segment by hand
    start_point1 = (int(W / 2), int(H / 2))
    end_point1 = (W, H)

    start_point2 = (int(W / 2), int(H / 2))
    end_point2 = (0, H - 250)

    mask1 = np.zeros_like(canny)
    cv.line(mask1, start_point1, end_point1, 255, 2)
    cv.fillPoly(mask1, [np.array([(0, 0), start_point1, end_point1, (W, 0)])], 255)
    canny[mask1 > 0] = 0

    mask2 = np.zeros_like(canny)
    cv.line(mask2, start_point2, end_point2, 255, 2)
    cv.fillPoly(mask2, [np.array([(W, 0), start_point2, end_point2, (0, 0)])], 255)
    canny[mask2 > 0] = 0

    mask_below = np.zeros_like(canny)
    cv.fillPoly(
        mask_below,
        [np.array([[0, H - 250], [int(W / 2), int(H / 2)], [W, H], [0, H]])],
        255,
    )
    canny[mask_below > 0] = 255
    # end segment by hand

    dpi = 100
    fig_size = (W / dpi, H / dpi)

    canny[canny == 255] = 1
    np.savetxt(f"{ROOT_TEXT}/segment_values.txt", canny, fmt="%d")

    res = np.multiply(gray, canny)
    plt.figure(figsize=fig_size, dpi=dpi)
    plt.imshow(res)
    plt.axis("off")
    plt.savefig(
        f"{ROOT_PICTURE}/segment_result.jpg", bbox_inches="tight", pad_inches=0, dpi=dpi
    )
    plt.close()
