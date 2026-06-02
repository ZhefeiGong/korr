"""Visualize AprilTag detection from three cameras."""
import argparse
import numpy as np
import cv2
from datetime import datetime

from furniture_bench.perception.realsense import RealsenseCam
from furniture_bench.perception.apriltag import AprilTag
from furniture_bench.utils.draw import draw_tags
from furniture_bench.config import config


def main():
    
    cam1 = RealsenseCam(
        config["camera"][1]["serial"],
        config["camera"]["color_img_size"],
        config["camera"]["depth_img_size"],
        config["camera"]["frame_rate"],
    )
    cam2 = RealsenseCam(
        config["camera"][2]["serial"],
        config["camera"]["color_img_size"],
        config["camera"]["depth_img_size"],
        config["camera"]["frame_rate"],
        None,
        disable_auto_exposure=True,
    )
    cam3 = RealsenseCam(
        config["camera"][3]["serial"],
        config["camera"]["color_img_size"],
        config["camera"]["depth_img_size"],
        config["camera"]["frame_rate"],
    )

    fps = 30
    frame_width = int(3840)
    frame_height = int(720)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 格式编码

    recording = False
    video_writer = None


    while True:
        color_img1, _ = cam1.get_frame()
        color_img2, _ = cam2.get_frame()
        color_img3, _ = cam3.get_frame()
        color_img = np.hstack([np.asanyarray(color_img1.get_data()).copy(), 
                               np.asanyarray(color_img2.get_data()).copy(), 
                               np.asanyarray(color_img3.get_data()).copy()])
        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Detected tags", color_img)
        key = cv2.waitKey(1)

        # begin
        if key & 0xFF == ord('b') and not recording:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = f"/home/dingpengxiang/jeffrey/korr_videos/record_{timestamp}.mp4"
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            recording = True
            print(f"Recording started... Saving to {output_path}")
        # stop
        elif key & 0xFF  == ord('e') and recording:
            recording = False
            video_writer.release()
            video_writer = None
            print("Recording stopped.")
        # write
        if recording and video_writer is not None:
            video_writer.write(color_img)
        # Out
        if key == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()


