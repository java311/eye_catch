import argparse
import pathlib
import numpy as np
import cv2
import time
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from PIL import Image, ImageOps

from face_detection import RetinaFace

from l2cs import select_device, draw_gaze, getArch, savePoint, Pipeline, render, loadPoint

CWD = pathlib.Path.cwd()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evaluation using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="gpu", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str) #default ResNet50

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    cam = args.cam_id
    # snapshot_path = args.snapshot

    gaze_pipeline = Pipeline(
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device = select_device(args.device, batch_size=1)
    )
     
    # cam = "D:\\tmp\\chuou_rinkan_videos\\station_entrance.mp4"
    cam = "D:\\faceswap\\micro_cherry.mp4"
    cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    # remove temporal save file
    if os.path.isfile("l2cs_tmp.json"):
        os.remove("l2cs_tmp.json")

    count = 0
    with torch.no_grad():
        while True:

            # Get frame
            success, frame = cap.read()    
            start_fps = time.time()  

            if not success:
                print("Finish processing")
                time.sleep(0.1)
                break
            else:
                print("Processing frame:" + str(count))
                count = count + 1

            # Process frame
            results = gaze_pipeline.step(frame)

            # Visualize output
            # save_frame(frame, results, "l2cs_tmp.json") 
            # frame = render(frame, results, "l2cs_tmp.json") 
            frame = savePoint(frame, results, "l2cs_tmp.json")
            results = None
            results = loadPoint("l2cs_tmp.json")
            frame = render(frame, results[-1]) 
            # frame = renderPoint(frame, results)
           
            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            success,frame = cap.read()  
    
