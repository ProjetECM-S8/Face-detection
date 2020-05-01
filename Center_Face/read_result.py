import cv2
import sys, datetime
from time import sleep
from centerface import CenterFace

import numpy as np

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)


def draw_boxes(frame, num, color=(0, 255, 0)):
    label=0
    file = open(str(num)+'.txt', "r")
    list_arr = file.readlines()
    l = len(list_arr)
    for i in range(l):
        list_arr[i] = list_arr[i].strip()
        list_arr[i] = list_arr[i].strip('[]')
        list_arr[i] = list_arr[i].split(", ")
    a = np.array(list_arr)
    a = a.astype(float)
    for i in range(len(a)):
        label=str(a[i][0])
        print(label)
        print(a[i][1])
        cv2.rectangle(frame, (int(a[i][1]), int(a[i][2])), (int((a[i][3])), int(a[i][4])), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(label), (int(a[i][1]), int(a[i][2])), font, 1, (0, 0, 255), 1)
    file.close()
    return frame



def run(event_interval=6):
    video_capture = cv2.VideoCapture('foule.mp4')

    # exit if video not opened
    if not video_capture.isOpened():
        print('Cannot open video')
        sys.exit()

    # read first frame
    ok, frame = video_capture.read()
    h, w = frame.shape[:2]
    if not ok:
        print('Error reading video')
        sys.exit()

    # init detection pipeline
    pipeline = Pipeline(event_interval=event_interval)

    # hot start detection
    # read some frames to get first detection
    faces = ()
    detected = False
    while not detected:
        _, frame = video_capture.read()
        faces, detected = pipeline.detect_and_track(frame, h, w, threshold=0.35)
        #print("hot start; ", faces, type(faces), "size: ", np.array(faces).size)

    draw_boxes(frame,0)

    ##
    ## main loop
    ##
    fps = 48  # 视频每秒24帧
    size = (1920, 1080)  # 需要转为视频的图片的尺寸
    # 可以使用cv2.resize()进行修改
    video = cv2.VideoWriter("result_foule.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
    num=0
    while True:
        # Capture frame-by-frame
        a, frame = video_capture.read()
        if a is not True:
            break


        color = BLUE
        num=num+1
        draw_boxes(frame, num, color)

        # Display the resulting frame
        video.write(frame)
        cv2.imshow('Video', frame)

        # quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", "-i", type=int,
                        action='store',
                        default=20,
                        help='Detection interval in seconds, default=6')

    args = parser.parse_args()
    run(args.interval)