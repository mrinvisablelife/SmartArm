import os
import cv2
import pyrealsense2 as rs
import numpy as np
import ultralytics
from ultralytics import YOLO
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

def Find_keypoints(results):
    for r in results:
        keypoints = r.keypoints.xy
        keypoints = keypoints.tolist()
        keypoints_conf = r.keypoints.conf
        keypoints_conf = keypoints_conf.tolist()
        print(keypoints_conf)
        if keypoints_conf:
            shoulder_conf = keypoints_conf[0][point_list[0]]
            elbow_conf = keypoints_conf[0][point_list[1]]
            wrist_conf = keypoints_conf[0][point_list[2]]
            conf= 0.4
            if (shoulder_conf > conf):
                shoulder = keypoints[0][point_list[0]]
            else:
                shoulder = None
            if (elbow_conf > conf):
                elbow = keypoints[0][point_list[1]]
            else:
                elbow = None
            if (wrist_conf > conf):
                wrist = keypoints[0][point_list[2]]
            else:
                wrist = None
            keypoints = [shoulder, elbow, wrist]
        else:
            keypoints = [None, None, None]

    return keypoints

def Draw_keyponit(image, keypoints):
    color_point = (0, 0, 255)
    color_line = (0, 255, 0)
    for point in [keypoints[0], keypoints[1], keypoints[2]]:
        if point is not None:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 5, color_point, -1)
        else:
            pass

    if keypoints[0] is not None and keypoints[1] is not None:
        cv2.line(image, (int(keypoints[0][0]), int(keypoints[0][1])), (int(keypoints[1][0]), int(keypoints[1][1])),color_line, 2)
    else:
        pass
    if keypoints[1] is not None and keypoints[2] is not None:
        cv2.line(image,(int(keypoints[1][0]), int(keypoints[1][1])),(int(keypoints[2][0]), int(keypoints[2][1])), color_line, 2)
    else:
        pass

    return image

def workout(model_path, pipeline, point_list, fps, up_angle, down_angle ,show):
    h, w, _ = get_image(pipeline)[1].shape
    out_path = "./run/" + "v1_2_1dep_results" + ".avi" # Output video path to run
    out_path1 = "./run/" + "v1_2_1col_results" + ".avi" # Output video path to run
    dep_video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    col_video_writer = cv2.VideoWriter(out_path1, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(model_path)

    play = 1 #video play control

    while True:
        dep_image, col_image = get_image(pipeline)
        dep_results = model(dep_image)
        col_results = model(col_image)
        keypoints_dep = Find_keypoints(dep_results)
        keypoints_col = Find_keypoints(col_results)
        dep_results = Draw_keyponit(dep_image, keypoints_dep)
        col_results = Draw_keyponit(col_image, keypoints_col)
        dep_video_writer.write(dep_results)
        col_video_writer.write(col_results)
        cv2.namedWindow('color frame', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('color frame', dep_results)
        cv2.namedWindow('depth frame', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('depth frame', col_results)
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