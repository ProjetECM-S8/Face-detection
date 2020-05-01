from imageSolver import *
from detection import *

detector = Detector(modelPath='./infer_model',USE_CUDA=False)
imgs,bboxes_pre = detector(imgList=['imgs/9.jpg', 'imgs/6.jpg'],
	confidence_threshold=0.3,nms_threshold=0.3)

for i,(img,bbox_pre) in enumerate(zip(imgs,bboxes_pre)):
    draw_bbox(img, bbox_pre, savePath=f'imgs/{i+1}_out.jpg')