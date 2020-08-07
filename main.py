# You may need to restart your runtime prior to this, to let your installation take effect

# Some basic setup:
# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
# import some common libraries

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle as pkl

# import utility function
from eyedk_utils import *

def run_pipline(video_path, num_frames=10, detection_mode = "BirdEye"):
    temp_dir = "temp_dir/" # the raw frames of the video will be stored here
    temp_dir_out = "temp_dir_out/" # the frames processed by yolo would be stored here
    process_out_path = "temp_dir_distance/" # the frame processed by the distance measuring algorithm would be stored here
    try:
        os.mkdir(temp_dir)
        os.mkdir(temp_dir_out)
        os.mkdir(process_out_path)
    except:
        print("folders already exist")
    for item in os.listdir(temp_dir):
        os.remove(temp_dir + item)
    # convert the video to frames
    cnt = video_to_frames(video_path, num_frames, temp_dir)
    print("frame conversion completed")
    # Select bottom right clockwise
    matrix, imgOutput = find_matrix("temp_dir/0.png")
    # perform detection on the video then output the sequence of detected images and the json file
    detect_with_yolo(out = temp_dir_out, source = temp_dir)
    print("All image processing completed")
    # convert the frames back to a video
    generate_video(temp_dir_out, cnt)
    print("detection video generated")
    json_dict = read_dict("temp_dir_out/json_out.txt")
    distance_metric_evaluation(json_dict, matrix, imgOutput, mode=detection_mode)
    generate_video(process_out_path, cnt)

if __name__ == "__main__":
    # Run on video
    run_pipline("data/videoplayback.mp4")

