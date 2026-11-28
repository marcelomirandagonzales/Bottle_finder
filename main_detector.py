import bottledetector_live as bd
import Camera_frames

import pyrealsense2 as rs

from ultralytics import YOLO
from ultralytics.engine.results import Results
from norfair import Detection
from typing import List
import numpy as np
import cv2


def main():

    model = bd.BottleDetector()

    pipeline = rs.pipeline()

    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense', color_image)

            nf_detections, pos = model.predict_frame(color_image, 0.5)
            print(pos)
            print(len(nf_detections))

            if len(nf_detections) > 0:
                print("Bottle detected!")

                try:

                    print("Stopping camera...")

                    pipeline.stop()

                    pipeline = rs.pipeline()
                    pipeline.start()

                    frames = pipeline.wait_for_frames()
                    depth_frame = frames.get_depth_frame()

                    if not depth_frame:
                        print("No depth frame available")
                        continue
                    
                    distances = []

                    for i in range(int(pos[0][0]), int(pos[1][0])):
                        for j in range(int(pos[0][1]), int(pos[1][1])):
                            dist = depth_frame.get_distance(i, j)
                            if dist > 0:
                                distances.append(dist)
                        else:
                            continue

                    print(f"The camera is facing an object {np.mean(distances):.5f} meters away", end="\r")

                    pipeline.stop()

                    pipeline = rs.pipeline()

                    config = rs.config()

                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

                    profile = pipeline.start(config)

                except Exception as e:
                    print(f"Error occurred while getting depth: {e}")


    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting camera...")
    main()