import cv2
import numpy as np
import os
import json
import math
import winsound
from .results import GazeResultContainer
from .moller2 import *

DISTANCE_TO_OBJECT = 1000  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm

def draw_gaze(a,b,c,d,image_in, pitchyaw, thickness=2, color=(0, 0, 255),sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = c
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out

def draw_bbox(frame: np.ndarray, bbox: np.ndarray, color: tuple):
    
    x_min=int(bbox[0])
    if x_min < 0:
        x_min = 0
    y_min=int(bbox[1])
    if y_min < 0:
        y_min = 0
    x_max=int(bbox[2])
    y_max=int(bbox[3])

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 1)

    return frame

def render(frame: np.ndarray, results: GazeResultContainer, filePath: str):

    savePoint(frame, results, filePath)

    return renderPoint(frame, results)

    # # Draw bounding boxes
    # for bbox in results.bboxes:
    #     frame = draw_bbox(frame, bbox)

    # # Draw Gaze
    # for i in range(results.pitch.shape[0]):

    #     bbox = results.bboxes[i]
    #     pitch = results.pitch[i]
    #     yaw = results.yaw[i]
        
    #     # Extract safe min and max of x,y
    #     x_min=int(bbox[0])
    #     if x_min < 0:
    #         x_min = 0
    #     y_min=int(bbox[1])
    #     if y_min < 0:
    #         y_min = 0
    #     x_max=int(bbox[2])
    #     y_max=int(bbox[3])

    #     # Compute sizes
    #     bbox_width = x_max - x_min
    #     bbox_height = y_max - y_min

    #     draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch,yaw),color=(0,0,255))

    # return frame

def savePoint(frame: np.ndarray, results: GazeResultContainer, filePath: str):
    
    # save into tmp file
    # with open(filePath, 'w', encoding='utf-8') as f:
    #     json.dump(results.to_json(), f, ensure_ascii=True, indent=4)
    
    cdir = os.getcwd()
    fpath =  os.path.join(cdir, filePath)

    a = []
    if os.path.isfile(fpath) == False:
        a.append(results.to_json())
        with open(fpath, mode='w') as f:
            f.write(json.dumps(a, indent=2))
    else:
        with open(fpath) as feedsjson:
            feeds = json.load(feedsjson)

        feeds.append(results.to_json())
        with open(fpath, mode='w') as f:
            f.write(json.dumps(feeds, indent=2))

    return frame

def yaw_pitch_to_vector(yaw, pitch):
    x = math.cos(yaw) * math.cos(pitch)
    y = math.sin(yaw) * math.cos(pitch)
    z = math.sin(pitch)
    
    return (x, y, z)

def draw_vector(image, yaw, pitch, length=100, thickness=2, color=(0, 255, 0)):
    # Get the center of the image
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    
    # Calculate the endpoint of the vector
    x, y, z = yaw_pitch_to_vector(yaw, pitch)
    endpoint = (int(center[0] + x * length), int(center[1] + y * length))
    
    # Draw the vector on the image
    cv2.arrowedLine(image, center, endpoint, color, thickness)
    
    return image

def playBeep():
    frequency = 2600  # Set Frequency To 2500 Hertz
    duration = 200  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def renderPoint(frame: np.ndarray, results: GazeResultContainer):

    # gaze do draw calcultions and variables
    image_height, image_width = frame.shape[:2]
    quadrants = [
        ("center", (int(image_width / 4), int(image_height / 4), int(image_width / 4 * 3), int(image_height / 4 * 3))),
        ("top_left", (0, 0, int(image_width / 2), int(image_height / 2))),
        ("top_right", (int(image_width / 2), 0, image_width, int(image_height / 2))),
        ("bottom_left", (0, int(image_height / 2), int(image_width / 2), image_height)),
        ("bottom_right", (int(image_width / 2), int(image_height / 2), image_width, image_height)),
    ]

    # Draw Gaze
    for i in range(results.pitch.shape[0]):

        bbox = results.bboxes[i]
        pitch = results.pitch[i]
        yaw = results.yaw[i]
        results.color[i] = (255,0,0)
        
        # Extract safe min and max of x,y
        x_min=int(bbox[0])
        if x_min < 0:
            x_min = 0
        y_min=int(bbox[1])
        if y_min < 0:
            y_min = 0
        x_max=int(bbox[2])
        y_max=int(bbox[3])

        # Compute sizes
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch,yaw),color=(255,0,0))

        # THIS CODE WORKS DO NOT DELETE IT !!!!!!!!!!!
        # draw a red circle on the screen (depend on the user looking spot)
        # length_per_pixel = HEIGHT_OF_HUMAN_FACE /  abs(bbox[1] - bbox[3])
        # length_per_pixel = 1

        # dx = -DISTANCE_TO_OBJECT * np.tan(results.pitch[i]) / length_per_pixel
        # # 100000000 is used to denote out of bounds
        # dx = dx if not np.isnan(dx) else 100000000
        # dy = -DISTANCE_TO_OBJECT * np.arccos(results.pitch[i]) * np.tan(results.yaw[i]) / length_per_pixel
        # dy = dy if not np.isnan(dy) else 100000000

        # # gaze_point = int(image_width / 2 + dx), int(image_height / 2  + dy)
        
        # gaze_point = min(int(image_width / 2 + dx), image_width), min(int(image_height / 2  + dy), image_height)

        # cv2.circle(frame, gaze_point, 10, (0, 0, 255), -1)
        # THIS CODE WORKS DO NOT DELETE IT !!!!!!!!!!!

        dv = yaw_pitch_to_vector(results.yaw[i], results.pitch[i])
        print (dv)

        # screen = [ Vec3(-1, -1, 0), Vec3(-1, 1, 0), Vec3(1, -1, 0),
        #          Vec3(1, 1, 0), Vec3(1, -1, 0), Vec3(-1, 1, 0)]
        # UNITS SEEMS TO BE 2 = 40 cm

        # CAMERA ON THE CENTER OF THE SCREEN (20 cm x 20cm)
        # screen = [ Vec3(0.200000, 0.200000, 1.000000), Vec3(-0.200000, -0.200000, 1.000000), Vec3(0.200000, -0.200000, 1.000000),
        #           Vec3(0.200000, 0.200000, 1.000000), Vec3(-0.200000, 0.200000, 1.000000), Vec3(-0.200000, -0.200000, 1.000000)] 
        
        # CAMERA ON TOP OF lg house screen (60 cm x 40cm)
        screen = [ Vec3(0.300000, 0.000000, 1.000000), Vec3(-0.300000, -0.400000, 1.000000), Vec3(0.300000, -0.400000, 1.000000),
                  Vec3(0.300000, 0.000000, 1.000000), Vec3(-0.300000, 0.000000, 1.000000), Vec3(-0.300000, -0.400000, 1.000000)] 

        # tuples = rays_triangles_intersection(np.array([0, 0, 1]), np.array(dv), np.array(screen))

        r = Ray()
        # TODO. It is necessary to add the user head coordinates here 
        r.orig = Vec3(0, -0.15, 2)   #(subject is 1 meter front the screen)
        r.direction = Vec3(dv[2], dv[1], -1)
        # print (r.direction)
        t = ray_triangle_intersect(r, screen[0],
                                      screen[1],
                                      screen[2])
        t2 = ray_triangle_intersect(r, screen[3],
                                      screen[4],
                                      screen[5])

        if t >= 0: 
            print ("hit") 
            playBeep()     
            results.color[i] = (0,255,0) 
        if t2 >= 0: 
            print ("hit")
            playBeep()
            results.color[i] = (0,255,0)
        # print (math.degrees(results.yaw[i]),math.degrees(results.pitch[i]))
        # frame = draw_vector(frame, results.yaw[i], results.pitch[i])
        # frame = render_3d_vector(frame, [int(image_width/2),int(image_height/2),0], dv, 50)

    # Draw bounding boxes
    index = 0
    for bbox in results.bboxes:
        frame = draw_bbox(frame, bbox, results.color[index])
        index = index + 1 

    return frame
