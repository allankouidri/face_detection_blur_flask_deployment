# Face detection and blur model - Flask deployment
This repository contains codes to detect and blur faces. 
The workflow uses the [ikomia api](https://github.com/Ikomia-dev/IkomiaApi) which is an open source tool allowing to create easily computer vision application. 
 We use [Kornia face detector](https://kornia.readthedocs.io/en/latest/applications/face_detection.html), an algo that I integrated into ikomia, it can be found in this [repository](https://github.com/Ikomia-hub/infer_face_detection_kornia)

## Install 
[Python>=3.7.0](https://www.python.org/downloads/release/python-370/) is required with requirements1.txt and requirements2.txt
```
$ git clone https://github.com/ultralytics/yolov5
$ cd face_detection_blur_flask_deployment
$ pip install -r requirements1.txt
$ pip install -r requirements2.txt
```
### Detection and blur from images
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

### Detection and blur from camera
To run the camera detection script:

```
python detect_from_cam.py
```
Press 'q' to quit.

### Detection and blur using Flask web framework



Project Structure
This project has four major parts :

model.py - This contains code fot our Machine Learning model to predict employee salaries absed on trainign data in 'hiring.csv' file.
app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
templates - This folder contains the HTML template to allow user to enter employee detail and displays the predicted employee salary.
Running the project
Ensure that you are in the project home directory. Create the machine learning model by running below command -
python model.py
This would create a serialized version of our model into a file model.pkl

Run app.py using below command to start Flask API
python app.py
By default, flask will run on port 5000.

Navigate to URL http://localhost:5000
You should be able to view the homepage as below : alt text

Enter valid numerical values in all 3 input boxes and hit Predict.

If everything goes well, you should be able to see the predcited salary vaule on the HTML page! alt text

You can also send direct POST requests to FLask API using Python's inbuilt request module Run the beow command to send the request with some pre-popuated values -
python request.py