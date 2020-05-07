# CenterFace face detection and KCF face tracking pipeline

This project implements CenterFace, a high-performance and small neural network architecture designed specifically for face detection. Once CenterFace detects image, the tracker KCF is activated to track the trajectory of face movement.

The CenterFace architecture is implemented in the file `centerface.py`. Its model required is stocked in `Center_Face\models\onnx\centerface.onnx`

### GPU Version with TensorRT

The Gpu version of detector is possible with file 'centerface_tsr.py'. 

Requirement:
`Python 3.6`, 
`Pycuda`, 
`Cuda 10.1`, 
`TensorRT 6.1.0.5`

You need to transfer model `.onnx` to `.tsr` with code `onnxtrans.py` first, then put it through `centerface_tsr.py`.


### Pipeline

1. Run detector, get boxes (Green)
2. Track boxes for each frame (Blue)
3. Update detector periodically or on-demand, and re-create trackers for each box. 

### Usage
Change Directory to `Center_face`

````
python tracking_centerface.py
````

Optionally use `-i` argument to set different detection intervals, in seconds. Default=20.

````
python tracking_centerface.py -i 3
````

The program saves the faces detected and tracked in txt format in the `Output` directory. Each txt file contains the face boxes `[No.Face, x,y,width,height]` in a frame.

To read and to load the boxes saved in the txt files in the original video

```
python read_result.py
```

