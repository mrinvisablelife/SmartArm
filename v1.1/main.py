import os
import cv2
import ultralytics
from ultralytics import solutions
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

def workout(model_path, camera, point_list, fps, up_angle, down_angle ,show):
    cap = cv2.VideoCapture(camera)
    assert cap.isOpened(), "Error reading"
    w, h = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = "./run/" + "v1_1" + ".avi" # Output video path to run
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

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

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = gym(im0)
        video_writer.write(results.plot_im)
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