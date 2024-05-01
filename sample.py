from l2cs import Pipeline, render
import cv2
import torch

gaze_pipeline = Pipeline(
    weights="models\\L2CSNet_gaze360.pkl",
    arch='ResNet50',
    device=torch.device('cpu') # or 'gpu'
)
 
# cap = cv2.VideoCapture("..\\data\\shinjuku.mp4")
# cap = cv2.VideoCapture("..\\data\\shinjuku.mp4")
# _, frame = cap.read()    

frame = cv2.imread("..\\data\\festival.png")

# Process frame and visualize
results = gaze_pipeline.step(frame)
frame = render(frame, results)