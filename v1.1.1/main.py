import os
import cv2
import ultralytics
from ultralytics import YOLO
ultralytics.checks()

def fpscounter(camera):
    cap = cv2.VideoCapture(camera)
    num_frames = 120
    print("Calculating FPS...")
    start = cv2.getTickCount()
    for _ in range(num_frames):
        ret, frame = cap.read()
        while not ret:
            ret, frame = cap.read()
    end = cv2.getTickCount()
    seconds = (end - start) / cv2.getTickFrequency()
    fps = int(num_frames / seconds)
    print(f"Estimated FPS: {fps}")
    return fps

def draw_arm(image, keypoints):
    color = (0, 0, 255)
    for point in [keypoints[0], keypoints[1], keypoints[2]]:
        if point is not None:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 5, color, -1)
        else:
            pass

    if keypoints[0] is not None and keypoints[1] is not None:
        cv2.line(image, (int(keypoints[0][0]), int(keypoints[0][1])), (int(keypoints[1][0]), int(keypoints[1][1])),color, 2)
    else:
        pass
    if keypoints[1] is not None and keypoints[2] is not None:
        cv2.line(image,(int(keypoints[1][0]), int(keypoints[1][1])),(int(keypoints[2][0]), int(keypoints[2][1])), color, 2)
    else:
        pass

    return image

def workout(model_path, camera, point_list, fps, up_angle, down_angle ,show):
    cap = cv2.VideoCapture(camera)
    assert cap.isOpened(), "Error reading"
    w, h = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = "./run/" + "v1_1_1" + ".avi" # Output video path to run
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(model_path)

    play = 1 #video play control

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = model(im0)
        for r in results:
            keypoints = r.keypoints.xy
            keypoints = keypoints.tolist()
            keypoints_conf = r.keypoints.conf
            keypoints_conf = keypoints_conf.tolist()
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

        im1 = draw_arm(im0, keypoints)
        cv2.namedWindow('YOLO POSE', cv2.WINDOW_AUTOSIZE)
        cv2.imshow("YOLO Pose", im1)
        video_writer.write(im1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # ESC or q key to exit
            break
        elif key == 27:
            break
        elif key == 13: # Enter key to pause
            play = play ^ 1 #line 38
        else:
            pass
    
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "./model/yolo11n-pose.pt"
    camera = 0
    point_list = [5, 7, 9]  # 關鍵點選擇https://github.com/ultralytics/docs/releases/download/0/keypoints-order-ultralytics-yolov8-pose.avif
    fps=fpscounter(camera)
    up_angle=160
    down_angle=20
    show=True
    workout(model_path, camera, point_list, fps, up_angle, down_angle, show)