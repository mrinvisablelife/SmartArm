import os
import cv2
import ultralytics
from ultralytics import solutions
ultralytics.checks()

def workout(model_path, video_part, point_list, up_angle=170, down_angle=0 ,show=True):
    cap = cv2.VideoCapture(video_part)

    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    out_path = "./run/" + os.path.splitext(os.path.basename(video_part))[0] + ".avi" # Output video path to run
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    gym = solutions.AIGym(
        model=model_path,
        show=show,
        up_angle=up_angle,
        down_angle=down_angle,
        line_width=2,
        kpts=point_list,
    )

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or processing is complete.")
            break
        results = gym(im0)
        video_writer.write(results.plot_im)

    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "./model/yolo11n-pose.pt"
    video_part = "./video/test2.mp4"
    point_list = [5, 7, 9]  # 關鍵點選擇https://github.com/ultralytics/docs/releases/download/0/keypoints-order-ultralytics-yolov8-pose.avif
    workout(model_path, video_part, point_list)