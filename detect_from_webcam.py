import os
import ikomia 
from ikomia.utils import ik
from ikomia.dataprocess import workflow
import numpy as np
import cv2

os.environ['IKOMIA_USER'] = "demo"
os.environ['IKOMIA_PWD'] = "jH4q72DApbRPa4k"

ikomia.authenticate()


def detect():
    """Dectect and blurr faces """
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


    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        image_np = np.array(frame)
        
        # Run inference 
        wf.run_on(image_np)
        
        # Get and show image
        img = wf.get_image(blurr_id, index=0)
        cv2.imshow('object detection (press \'q\' to quit)', img)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    detect()