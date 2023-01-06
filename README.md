# Face detection and blur model - Flask deployment
This repository contains codes to detect and blur faces. 
The workflow uses the [ikomia api](https://github.com/Ikomia-dev/IkomiaApi) which is an open source tool allowing to create easily computer vision application. 
 We use [Kornia face detector](https://kornia.readthedocs.io/en/latest/applications/face_detection.html), an algo that I integrated into ikomia, it can be found in this [repository](https://github.com/Ikomia-hub/infer_face_detection_kornia).

## Install 
Python 3.7, 3.8 or 3.9 is required with requirements1.txt and requirements2.txt
```
$ git clone https://github.com/ultralytics/yolov5
$ cd face_detection_blur_flask_deployment
$ pip install -r requirements1.txt
$ pip install -r requirements2.txt
```
## Inference 

### 1- Detection and blur from images
Before running the inference, make sur there are some test images in the following folder

```
images\detect_image
```

This script will run the workflow on all the images from the detect_image folder and save the results in:

```
images\detect_res
```

Detection and blur can be done running the command: 

```
python detect_from_image.py  --images_folder images\detect_image --output_path images\detect_res
``` 

### 2- Detection and blur from camera
To run the camera detection script:

```
python detect_from_cam.py
```
Press 'q' to quit.


### 3- Detection and blur using Flask web framework
Start the Flask app
```
python app.py 
```
By default, flask will run on port 5000.
Navigate to URL http://localhost:5000

If everything goes well, you should be able to see the live camera and blurred faces!