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

def run_pipline(video_path):
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
    num_frames = 10
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
    distance_metric_evaluation(json_dict, matrix, imgOutput, mode="BirdEye")
    generate_video(process_out_path, cnt)

if __name__ == "__main__":
    # Run on video
    run_pipline("data/videoplayback.mp4")

    setup_logger()
    video_path = "data/street cam.mp4"
    model_path = "model/model_final_721ade (1).pkl"
    config_path = "configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    num_frames = 300

    # loading model and config from the downloaded files
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    # Demo on first frame
    img = cv2.imread("frames/0.png")

    # pass to the model
    outputs = predictor(img)['instances']
    coord = outputs["pred_boxes"]
    # Use `Visualizer` to draw the predictions on the image.
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image())
    plt.show()

