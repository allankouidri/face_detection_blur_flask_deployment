'''
This script detects all images in a specified folder and outputs them in another folder
Example:
python detect_from_image.py  --images_folder images\detect_image --output_path images\detect_res
'''
import os
import ikomia 
from ikomia.utils import ik
from ikomia.dataprocess import workflow
import numpy as np
import cv2
import argparse
import glob

os.environ['IKOMIA_USER'] = "demo"
os.environ['IKOMIA_PWD'] = "jH4q72DApbRPa4k"

ikomia.authenticate()


def detect(test_images, detect_res):
    # Create detection results folder
    if not os.path.exists(detect_res):
        os.makedirs(detect_res)

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

    # Define path to images and grab all image filenames
    images = glob.glob(test_images + '/*')
    
    # Loop over every image and perform detection
    for image_path in images: 
        # Loading image into python
        img = cv2.imread(image_path)
        image_np = np.array(img)
        
        # Run inference 
        wf.run_on(image_np)
        
        # Get and show image
        img = wf.get_image(blurr_id, index=0)
        
        filename=image_path.replace("\\","/")
        image_name = os.path.join(detect_res,filename.split('/')[-1])
        
        saved_img = cv2.imwrite(image_name ,img)
        
        
def main(args):
    '''
    python detect_from_image.py  
    --images_folder Tensorflow\workspace\images\detect_image 
    --output_path Tensorflow\workspace\images\detect_res
    '''
    TEST_IMAGE_PATH = args['images_folder']
    DETECT_RES_PATH = args['output_path']
    
    detect(test_images=TEST_IMAGE_PATH, detect_res=DETECT_RES_PATH)

if __name__ == '__main__':
            # create parser and handle arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--images_folder', required=True, help='path\\to\\image\\folder')
        parser.add_argument('--output_path', required=True, help='path\\to\\inference\results\\folder')

        args = vars(parser.parse_args())
        
        main(args)