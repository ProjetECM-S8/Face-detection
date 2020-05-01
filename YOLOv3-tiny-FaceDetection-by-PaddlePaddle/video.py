import os
import cv2
import glob
import numpy as np
from imageSolver import *
from detection import *
from timeit import default_timer as timer
import sys, datetime

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

detector = Detector(modelPath='./infer_model',USE_CUDA=True)

""" functions """
def DetectVideo(image):

  imgs,bboxes_pre = detector(imgList=[image], confidence_threshold=0.3,nms_threshold=0.3)
  faces = []
  for i,(img,bbox_pre) in enumerate(zip(imgs,bboxes_pre)): 
    """ bbox_pre is the list of faces on the single image"""
    for bbox in bbox_pre:
      faces.append(bbox[:4])
   
  return faces 

def draw_boxes(frame, boxes, color=(0,255,0)):
    for (x1,y1,x2,y2) in boxes:
        frame = cv2.rectangle(frame, ( int(round(x1)),int(round(y1)) ), ( int(round(x2)),int(round(y2)) ), color, 2)
    return frame

def RotateAntiClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip( trans_img, 0 )
    return new_img

def process_image(img):
    size = img.shape
    h, w = size[0], size[1]
    #长边缩放为min_side 
    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv2.resize(img, (new_w, new_h))
    # 填充至min_side * min_side
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    elif new_h % 2 != 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    elif new_h % 2 == 0 and new_w % 2 == 0:
        top, bottom, left, right = (min_side-new_h)/2, (min_side-new_h)/2, (min_side-new_w)/2, (min_side-new_w)/2
    else:
        top, bottom, left, right = (min_side-new_h)/2 + 1, (min_side-new_h)/2, (min_side-new_w)/2 + 1, (min_side-new_w)/2
    pad_img = cv2.copyMakeBorder(resize_img, int(top), int(bottom),int(left),int(right),cv2.BORDER_CONSTANT, value=[0,0,0]) #从图像边界向上,下,左,右扩的像素数目
    #print pad_img.shape
    #cv2.imwrite("after-" + os.path.basename(filename), pad_img)
    return pad_img

class Controller():
    
    def __init__(self, event_interval=0):
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

class FaceTracker():
    
    def __init__(self, frame, face):
        (x,y,w,h) = face
        self.face = (x,y,w,h)
        # Arbitrarily picked KCF tracking
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, self.face)
    
    def update(self, frame):
        _, self.face = self.tracker.update(frame)
        return self.face

class Pipeline():

    def __init__(self, event_interval=0):
        self.controller = Controller(event_interval=event_interval)    
        # self.detector = DetectVideo()
        self.trackers = []
    
    def detect_and_track(self, frame):
        # get faces 
        faces = DetectVideo(frame)
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
    
    def boxes_for_frame(self, frame):
        if self.controller.trigger():
            return self.detect_and_track(frame)
        else:
            return self.detect_and_track(frame)


cap = cv2.VideoCapture("/home/jingyu/Documents/yolo/YOLOv3-tiny-FaceDetection-by-PaddlePaddle/video/Manif.mp4") 
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output_d.avi',fourcc, 24.0, (416,416))

# /Manif.mp4
accum_time = 0
curr_fps = 0
fps = "FPS: ??"
prev_time = timer()
start_time = timer()
bbox_video=[]
num = 0
event_interval = 5
min_side=416 #input size for yolo

if not cap.isOpened():
  print('Cannot open video')
  sys.exit()
    
pipeline = Pipeline(event_interval=event_interval)

# hot start detection
# read some frames to get first detection
faces = ()
detected = False
while not detected:
  _, frame = cap.read()
  frame = process_image(frame)
  if _:
    faces, detected = pipeline.detect_and_track(frame)
    print("hot start; ", faces, type(faces), "size: ", np.array(faces).size)
  else:
    break

draw_boxes(frame, faces)

while (cap.isOpened()):

  ret, frame = cap.read()
  if ret:
    """detect faces and save bboxes"""
    frame = process_image(frame)

    boxes, detected_new = pipeline.boxes_for_frame(frame)
    # logging
    state = "DETECTOR" if detected_new else "TRACKING"
    print("[%s] boxes: %s" % (state, boxes))

    # update screen
    color = GREEN if detected_new else BLUE
    draw_boxes(frame, boxes, color)

    
    # bbox_video.append(bbox)

    """use saved bboxes"""
    # bbox_video = np.load("video_1.npy",allow_pickle=True)
    # bbox_video = bbox_video.tolist()
    # bbox = bbox_video[num]
    num+=1
    # frame = process_image(frame)
    # frame = draw_bbox(frame, bbox)
    """                           """

    """calculate time and fps"""
    curr_time = timer()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    accum_time = accum_time + exec_time
    total_time = curr_time - start_time
    curr_fps = curr_fps + 1
    if accum_time > 1:
        accum_time = accum_time - 1
        fps = curr_fps
        curr_fps = 0

    info = [
          ('FPS', '{}'.format(fps)),
          ('time', '{}'.format(int(total_time))),
          ('resolution', '{}'.format(frame.shape)),
          ('num of frames', '{}'.format(num))

      ]
    for (i, (txt, val)) in enumerate(info):
      text = '{}: {}'.format(txt, val)
      cv2.putText(frame, text, (10, (i * 20) + 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 175, 0), )

    out.write(frame)
    # cv2.namedWindow("face", cv2.WINDOW_NORMAL)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  else:
    break

# bbox_video1_a = np.array(bbox_video)
# np.save("video_1.npy",bbox_video1_a)
cap.release()
out.release()
cv2.destroyAllWindows()