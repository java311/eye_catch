#mivolo imports
import argparse
import os
import torch
import cv2
import logging
import yt_dlp

from mivolo.data.data_reader import InputType, get_all_files, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging
_logger = logging.getLogger("inference")

# l2cs-net imports 
import time 
import pathlib

from PIL import Image
from PIL import Image, ImageOps

from face_detection import RetinaFace
from l2cs import select_device, draw_gaze, getArch, savePoint, Pipeline, render, loadPoint
CWD = pathlib.Path.cwd()

def get_direct_video_url(video_url):
    ydl_opts = {
        "format": "bestvideo",
        "quiet": True,  # Suppress terminal output (remove this line if you want to see the log)
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)

        if "url" in info_dict:
            direct_url = info_dict["url"]
            resolution = (info_dict["width"], info_dict["height"])
            fps = info_dict["fps"]
            yid = info_dict["id"]
            return direct_url, resolution, fps, yid

    return None, None, None, None

def get_local_video_info(vid_uri):
    cap = cv2.VideoCapture(vid_uri)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source {vid_uri}")
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, res, fps

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MiVOLO Inference")
    parser.add_argument("--input", type=str, default=None, required=True, help="image file or folder with images")
    parser.add_argument("--output", type=str, default=None, required=True, help="folder for output results")
    parser.add_argument("--detector-weights", type=str, default=None, required=True, help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", default="", type=str, required=True, help="path to mivolo checkpoint")

    parser.add_argument(
        "--with-persons", action="store_true", default=False, help="If set model will run with persons, if available"
    )
    parser.add_argument(
        "--disable-faces", action="store_true", default=False, help="If set model will use only persons if available"
    )

    parser.add_argument("--draw", action="store_true", default=False, help="If set, resulted images will be drawn")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")

    return parser

def process_eye_gaze(cap):

    gaze_pipeline = Pipeline(
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device = select_device("gpu", batch_size=1)
    )

    # remove temporal save file
    if os.path.isfile("l2cs_tmp.json"):
        os.remove("l2cs_tmp.json")

    count = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
                print("Processing frame:" + str(count) + " of:"+ str(length) + " " + str((count*length)/100.0) + "%")
                count = count + 1

            # Process frame
            results = gaze_pipeline.step(frame)

            # Visualize output
            # save_frame(frame, results, "l2cs_tmp.json") 
            # frame = render(frame, results, "l2cs_tmp.json") 
            savePoint(frame, results, "l2cs_tmp.json")
            
            # results = None
            # results = loadPoint("l2cs_tmp.json")
            # frame = render(frame, results[-1]) 
            # frame = renderPoint(frame, results)
           
            # myFPS = 1.0 / (time.time() - start_fps)
            # cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            # cv2.imshow("Demo",frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # success,frame = cap.read()

        print("l2CS Finished")

# render the l2cs-net json results into a new xvid video file
def render_eye_gaze(minovo_output_path, final_filepath):
    # for video input
    cap = cv2.VideoCapture(minovo_output_path)

    # for video output
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(final_filepath, fourcc, fps, res)

    count = 0
    results = loadPoint("l2cs_tmp.json")
    while True:
        success, frame = cap.read()    
        frame = render(frame, results[count])
        count = count + 1
        
        out.write(frame)
        cv2.imshow("Final Render",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)

    predictor = Predictor(args, verbose=True)

    input_type = get_input_type(args.input)

    if input_type == InputType.Video or input_type == InputType.VideoStream:
        if not args.draw:
            raise ValueError("Video processing is only supported with --draw flag. No other way to visualize results.")

        if "youtube" in args.input:
            args.input, res, fps, yid = get_direct_video_url(args.input)
            if not args.input:
                raise ValueError(f"Failed to get direct video url {args.input}")
            outfilename = os.path.join(args.output, f"out_{yid}.avi")
        else:
            bname = os.path.splitext(os.path.basename(args.input))[0]
            outfilename = os.path.join(args.output, f"out_{bname}.avi")
            cap, res, fps = get_local_video_info(args.input)

        if args.draw:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(outfilename, fourcc, fps, res)
            _logger.info(f"Saving result to {outfilename}..")

        # MiVOLO prediction (video)
        for (detected_objects_history, frame) in predictor.recognize_video(args.input):
            if args.draw:
                out.write(frame)

        #l2cs predictor and save results on file
        # cap, res, fps = get_local_video_info(args.input)
        # process_eye_gaze(cap)
        bname = os.path.splitext(os.path.basename(args.input))[0]
        outfilename = os.path.join(args.output, f"out_{bname}.avi")
        final_filename = os.path.join(args.output, f"final_{bname}.avi")
        render_eye_gaze(outfilename, final_filename)

    elif input_type == InputType.Image:
        image_files = get_all_files(args.input) if os.path.isdir(args.input) else [args.input]

        for img_p in image_files:

            img = cv2.imread(img_p)
            detected_objects, out_im = predictor.recognize(img)

            if args.draw:
                bname = os.path.splitext(os.path.basename(img_p))[0]
                filename = os.path.join(args.output, f"out_{bname}.jpg")
                cv2.imwrite(filename, out_im)
                _logger.info(f"Saved result to {filename}")

if __name__ == "__main__":
    main()