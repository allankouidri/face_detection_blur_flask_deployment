import os
import ikomia 
from ikomia.utils import ik
from ikomia.dataprocess import workflow
import numpy as np
from flask import Flask, render_template, Response
import cv2
import numpy as np

app=Flask(__name__)
camera = cv2.VideoCapture(0)

os.environ['IKOMIA_USER'] = "demo"
os.environ['IKOMIA_PWD'] = "jH4q72DApbRPa4k"
ikomia.authenticate()

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Init your workflow
            wf = workflow.create("Face detection and blur")

            # Add kornia face detection task
            face_dect_id, face_dect = wf.add_task("infer_face_detection_kornia")

            # Add OpenCV Gaussian blur task
            blurr_id, blurr = wf.add_task(ik.ocv_gaussian_blur)
            
            # Set parameters
            blurr_params = {
                ik.ocv_gaussian_blur_param.sigmaX: 45.0,
                ik.ocv_gaussian_blur_param.sigmaY: 45.0,
                ik.ocv_gaussian_blur_param.borderType: 2,
                ik.ocv_gaussian_blur_param.sizeX: 29,
                ik.ocv_gaussian_blur_param.sizeY: 29,
            }
                
            wf.set_parameters(task_id=blurr_id, params=blurr_params)
            
            # Connect tasks
            wf.connect_tasks(wf.getRootID(), face_dect_id)
            wf.connect_tasks(face_dect_id, blurr_id)
            
            image_np = np.array(frame)
            
            # Run inference 
            wf.run_on(image_np)
            
            # Get and show image
            frame = wf.get_image(blurr_id, index=0)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)