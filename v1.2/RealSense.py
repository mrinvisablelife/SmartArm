import os
import cv2
import pyrealsense2 as rs
import numpy as np
import ultralytics
from ultralytics import solutions
ultralytics.checks()

def fpscounter(pipeline, config):
    pipeline.start(config)
    rs.align(rs.stream.color)
    num_frames = 120
    print("Calculating FPS...")
    start = cv2.getTickCount()
    for _ in range(num_frames):
        frames = pipeline.wait_for_frames()
        while not frames:
            frames = pipeline.wait_for_frames()
    end = cv2.getTickCount()
    seconds = (end - start) / cv2.getTickFrequency()
    fps = int(num_frames / seconds)
    print(f"Estimated FPS: {fps}")
    return fps

def Realsense():
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480,  rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480,  rs.format.y8, 30)

    return pipeline, config

def get_image(pipeline):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    
    return depth_image, color_image

def workout(model_path, pipeline, point_list, fps, up_angle, down_angle ,show):
    h, w, _ = get_image(pipeline)[1].shape
    out_path = "./run/" + "v1_2dep_results" + ".avi" # Output video path to run
    out_path1 = "./run/" + "v1_2col_results" + ".avi" # Output video path to run
    dep_video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    col_video_writer = cv2.VideoWriter(out_path1, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    gym = solutions.AIGym(
        model=model_path,
        show=show,
        up_angle=up_angle,
        down_angle=down_angle,
        line_width=2,
        kpts=point_list,
        fps=fps,
    )

    play = 1 #video play control

    while True:
        depth_image, color_image = get_image(pipeline)
        dep_results = gym(depth_image)
        col_results = gym(color_image)
        dep_video_writer.write(dep_results.plot_im)
        col_video_writer.write(col_results.plot_im)
        cv2.namedWindow('color frame', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('color frame', dep_results.plot_im)
        cv2.namedWindow('depth frame', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('depth frame', col_results.plot_im)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # ESC or q key to exit
            break
        elif key == 27:
            break
        elif key == 13: # Enter key to pause
            play = play ^ 1 #line 38
        else:
            pass
    
    dep_video_writer.release()
    col_video_writer.release()
    cv2.destroyAllWindows()
    pipeline.stop()

if __name__ == "__main__":
    model_path = "./model/yolo11n-pose.pt"
    point_list = [5, 7, 9]  # 關鍵點選擇https://github.com/ultralytics/docs/releases/download/0/keypoints-order-ultralytics-yolov8-pose.avif
    pipeline, config = Realsense()
    fps=fpscounter(pipeline, config)
    up_angle=160
    down_angle=20
    show=False
    workout(model_path, pipeline, point_list, fps, up_angle, down_angle, show)