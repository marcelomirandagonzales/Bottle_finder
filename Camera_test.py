import pyrealsense2 as rs

pipeline = rs.pipeline()
print("Starting camera...")
pipeline.start() # Start streaming
print("test")

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        width, height = depth_frame.get_width(), depth_frame.get_height()
        dist = depth_frame.get_distance(width // 2, height // 2)
        print(f"The camera is facing an object {dist:.5f} meters away", end="\r")

finally:
    pipeline.stop() # Stop streaming

