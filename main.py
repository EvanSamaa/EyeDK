# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
# import some common libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from eyedk_utils import *
import pickle as pkl

def run_pipline(video_path):
    temp_dir = "temp_dir/"
    num_frames = 300

    cnt = video_to_frames(video_path, num_frames, temp_dir)

if __name__ == "__main__":
    setup_logger()
    video_path = "data/street cam.mp4"
    model_path = "model/model_final_721ade (1).pkl"
    config_path = "configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
    num_frames = 300

    # video_to_frames(video_path, num_frames)
    # loading model and config from the downloaded files
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    img = cv2.imread("frames/0.png")

    # pass to the model
    outputs = predictor(img)['instances']
    coord = outputs["pred_boxes"]
    # Use `Visualizer` to draw the predictions on the image.
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image())
    plt.show()

