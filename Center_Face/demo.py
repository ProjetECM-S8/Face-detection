import cv2
import scipy.io as sio
import os
from centerface import CenterFace
import time
import pandas as pd
import csv
def camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    centerface = CenterFace()
    while True:
        ret, frame = cap.read()
        dets, lms = centerface(frame, h, w, threshold=0.35)
        for det in dets:
            boxes, score = det[:4], det[4]
            cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        for lm in lms:
            for i in range(0, 5):
                cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
        cv2.imshow('out', frame)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def video():
    cap = cv2.VideoCapture('Manif.mp4')
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    centerface = CenterFace()
    fps = 48  # 视频每秒24帧
    size = (1920, 1080)  # 需要转为视频的图片的尺寸
    # 可以使用cv2.resize()进行修改
    video = cv2.VideoWriter("result01.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
   

    
    cnt_frame = 1


    output_path = "./Output/Manifcsv.csv"
    file = open(output_path,"w")
    writer = csv.writer(file)
    writer.writerow(["Frame","Faces list"])


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        dets, lms = centerface(frame, h, w, threshold=0.35)
        face_frame = []
        for det in dets:
            boxes, score = det[:4], det[4]
            face_frame.append(boxes)
            cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        #for lm in lms:
         #   for i in range(0, 5):
          #      cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
        writer.writerow([cnt_frame, face_frame])
        

        cnt_frame += 1
        video.write(frame)
        cv2.imshow('out', frame)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #dataframe = pd.DataFrame(columns = cnt_frame, data = face_list)
    #dataframe.to_csv('./Output/Manifcsv.csv',encoding = 'gbk')
    cap.release()


def test_image():
    frame = cv2.imread('000388.jpg')
    h, w = frame.shape[:2]
    landmarks = True
    centerface = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerface(frame, h, w, threshold=0.35)
    else:
        dets = centerface(frame, threshold=0.35)

    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    if landmarks:
        for lm in lms:
            for i in range(0, 5):
                cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
    cv2.imshow('out', frame)
    cv2.waitKey(0)


def test_image_tensorrt():
    frame = cv2.imread('000388.jpg')
    h, w = 480, 640  # must be 480* 640
    landmarks = True
    centerface = CenterFace(landmarks=landmarks, backend="tensorrt")
    if landmarks:
        dets, lms = centerface(frame, h, w, threshold=0.35)
    else:
        dets = centerface(frame, threshold=0.35)

    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    if landmarks:
        for lm in lms:
            for i in range(0, 5):
                cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
    cv2.imshow('out', frame)
    cv2.waitKey(0)


def test_widerface():
    Path = 'widerface/WIDER_val/images/'
    wider_face_mat = sio.loadmat('widerface/wider_face_split/wider_face_val.mat')
    event_list = wider_face_mat['event_list']
    file_list = wider_face_mat['file_list']
    save_path = 'save_out/'

    for index, event in enumerate(event_list):
        file_list_item = file_list[index][0]
        im_dir = event[0][0]
        # print(save_path + im_dir)
        if not os.path.exists(save_path + im_dir):
            os.makedirs(save_path + im_dir)
        landmarks = True
        centerface = CenterFace(landmarks=landmarks)
        for num, file in enumerate(file_list_item):
            im_name = file[0][0]
            zip_name = '%s/%s.jpg' % (im_dir, im_name)
            print(os.path.join(Path, zip_name))
            img = cv2.imread(os.path.join(Path, zip_name))
            h, w = img.shape[:2]
            if landmarks:
                dets, lms = centerface(img, h, w, threshold=0.05)
            else:
                dets = centerface(img, threshold=0.05)
            f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
            f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            f.write('{:d}\n'.format(len(dets)))
            for b in dets:
                x1, y1, x2, y2, s = b
                f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            f.close()
            print('event:%d num:%d' % (index + 1, num + 1))


if __name__ == '__main__':
     #camera()
    start = time.clock()
    video()
    end=time.clock()
    print("time",end-start)
    #test_image()
    # test_widerface()
