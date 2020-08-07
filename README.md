# EyeDK

## Demo

![VOLOv5](https://i.imgur.com/wcOoHxE.gif)

Detection on custom dataset with heatmap

## Code Setup

## Model Benchmark Test
### Colab Notebooks
[YOLOv5 baseline model benchmark](https://colab.research.google.com/drive/1DH1l-Dfnnta0Lb58YEc_PgAs0kwXP5zy?usp=sharing)

[Detectron2 baseline mode benchmark](https://colab.research.google.com/drive/1Mvs5pGpYEKoq2EHxQS8eJlJdgRlqSNPb?usp=sharing)

### Benchmark Dataset
[Multi-camera pedestrians video](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/) (Using)

[Caltech Pedestrian Detection Benchmark](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/)

[Joint Attention in Autonomous Driving JAAD Dataset](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) (Using)

### Benchmark Results
**EPFL dataset video 4p-c0**

*For VOLOv5*

![VOLOv5](https://i.imgur.com/nuWIslz.gif)

*For Detectron2*

![Detectron2](https://i.imgur.com/YeRtNxh.gif)


## Literature Resources
### Object Detection Models Review
[Responding to the Controversy about YOLOv5](https://blog.roboflow.ai/yolov4-versus-yolov5/)
This post compare between YOLO4 & YOLO5 performace on inference speed, model size and accuracy. Since paper for YOLO5 is not found, may be a good source for identifying YOLO5 performance in all

[Object Detection and Tracking in 2020](https://blog.netcetera.com/object-detection-and-tracking-in-2020-f10fb6ff9af3)

YOLO vs R-CNN/Fast R-CNN/Faster R-CNN is more of an apples to apples comparison (YOLO is an object detector, and Mask R-CNN is for object detection+segmentation).
YOLO is easier to implement due to its single stage architecture. Faster inference times and end-to-end training also means it'll be faster to train.

[Object Detection on COCO test-dev](https://paperswithcode.com/sota/object-detection-on-coco)

[Object Detection on COCO minival](https://paperswithcode.com/sota/object-detection-on-coco-minival)

### High Lighted Works
[A social distancing detector using a Tensorflow object detection model, Python and OpenCV(MobileNet)](https://towardsdatascience.com/a-social-distancing-detector-using-a-tensorflow-object-detection-model-python-and-opencv-4450a431238)

This post details the *bird's view conversion process* in detection, which uses `getPerspectiveTransform` and `warpPerspective` to transform the region of interest inside predefined four points. Similar approach is also adapted in [Bird's Eye View Transformation](https://nikolasent.github.io/opencv/2017/05/07/Bird's-Eye-View-Transformation.html)
### Related Works
[OpenCV Social Distancing Detector(With YOLO)](https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/)

[Social Distance Detector with Python, YOLOv4, Darknet, and OpenCV](https://heartbeat.fritz.ai/social-distance-detector-with-python-yolov4-darknet-and-opencv-62e66c15c2a4)

[Social Distancing Detector with YOLOv3 post](https://towardsdatascience.com/covid-19-ai-enabled-social-distancing-detector-using-opencv-ea2abd827d34)

[Social Distancing Detector with YOLOv3 code](https://github.com/mk-gurucharan/Social-Distancing-Detector)

[Your Social Distancing Detection Tool: How to Build One using your Deep Learning Skills](https://www.analyticsvidhya.com/blog/2020/05/social-distancing-detection-tool-deep-learning/)
### Bird's Eye View Conversion
[Bird's Eye View Transformation](https://nikolasent.github.io/opencv/2017/05/07/Bird's-Eye-View-Transformation.html)

[Robust lane finding techniques using computer vision — Python & Open CV](https://medium.com/@vamsiramakrishnan/robust-lane-finding-using-python-open-cv-63eb66fa2616)

[A Geometric Approach to Obtain a Bird’s Eye View from an Image](https://www.groundai.com/project/a-geometric-approach-to-obtain-a-birds-eye-view-from-an-image/1)


