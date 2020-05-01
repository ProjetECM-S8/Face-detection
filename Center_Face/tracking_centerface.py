import cv2
import sys, datetime
from time import sleep
from centerface import CenterFace

import numpy as np

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)


def draw_boxes(frame, boxes, num, color=(0, 255, 0)):
    label=0
    with open(str(num)+'.txt', 'a') as file_handle:  # 
        for (x, y, w, h) in boxes:
            label=label+1
            cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), color, 2)
            result2txt=str([label,x,y,w,h])
            file_handle.write(result2txt)  # 
            file_handle.write('\n')
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(label), (int(x), int(y)), font, 1, (0, 0, 255), 1)
        file_handle.close()
        return frame


def resize_image(image, size_limit=500.0):
    max_size = max(image.shape[0], image.shape[1])
    if max_size > size_limit:
        scale = size_limit / max_size
        _img = cv2.resize(image, None, fx=scale, fy=scale)
        return _img
    return image


class FaceTracker():

    def __init__(self, frame, face):
        (x, y, w, h) = face
        self.face = (x, y, w, h)
        # Arbitrarily picked KCF tracking
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, self.face)

    def update(self, frame):
        _, self.face = self.tracker.update(frame)
        return self.face


class Controller():

    def __init__(self, event_interval=6):
        self.event_interval = event_interval
        self.last_event = datetime.datetime.now()

    def trigger(self):
        """Return True if should trigger event"""
        return self.get_seconds_since() > self.event_interval

    def get_seconds_since(self):
        current = datetime.datetime.now()
        seconds = (current - self.last_event).seconds
        return seconds

    def reset(self):
        self.last_event = datetime.datetime.now()


class Pipeline():

    def __init__(self, event_interval=6):
        self.controller = Controller(event_interval=event_interval)
        self.detector = CenterFace()
        self.trackers = []

    def detect_and_track(self, frame, h, w, threshold=0.35):
        # get faces
        faces=[]
        faces1,lms = self.detector(frame, h, w, threshold=0.35)
        for det in faces1:
            boxes, score = det[:4], det[4]
            faces.append(boxes)

        # reset timer
        self.controller.reset()

        # get trackers
        self.trackers = [FaceTracker(frame, face) for face in faces]

        # return state = True for new boxes
        # if no faces detected, faces will be a tuple.
        new = type(faces) is not tuple

        return faces, new

    def track(self, frame):
        boxes = [t.update(frame) for t in self.trackers]
        # return state = False for existing boxes only
        return boxes, False

    def boxes_for_frame(self, frame, h, w, threshold=0.35):
        if self.controller.trigger():
            return self.detect_and_track(frame, h, w, threshold=0.35)
        else:
            return self.track(frame)


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

    draw_boxes(frame, faces,0)

    ##
    ## main loop
    ##
    fps = 48  # 
    size = (1920, 1080)  # 
    #
    video = cv2.VideoWriter("result_foule.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
    num=0
    while True:
        # Capture frame-by-frame
        _, frame = video_capture.read()

        # update pipeline
        boxes, detected_new = pipeline.boxes_for_frame(frame, h, w, threshold=0.35)

        # logging
        state = "DETECTOR" if detected_new else "TRACKING"
        #print("[%s] boxes: %s" % (state, boxes))

        # update screen
        color = GREEN if detected_new else BLUE
        num=num+1
        draw_boxes(frame, boxes, num, color)

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
                        help='Detection interval in seconds, default=20')

    args = parser.parse_args()
    run(args.interval)
