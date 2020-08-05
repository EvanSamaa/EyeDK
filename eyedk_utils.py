from scipy.spatial import distance
from detect import detect
import torch as torch
import imageio
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import imutils
import os
import re
from PIL import Image

################################ open cv functions ################################
def mid_point(img, person, idx):
    # get the coordinates
    x1, y1, x2, y2 = person[idx]
    _ = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # compute bottom center of bbox
    x_mid = int((x1 + x2) / 2)
    y_mid = int(y2)
    mid = (x_mid, y_mid)

    _ = cv2.circle(img, mid, 5, (255, 0, 0), -1)
    cv2.putText(img, str(idx), mid, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return mid
################################ Scipy functions ################################
def compute_distance(midpoints,num):
  dist = np.zeros((num,num))
  for i in range(num):
    for j in range(i+1,num):
      if i!=j:
        dst = distance.euclidean(midpoints[i], midpoints[j])
        dist[i][j]=dst
  return dist
def find_closest(dist,num,thresh):
  p1=[]
  p2=[]
  d=[]
  for i in range(num):
    for j in range(i,num):
      if( (i!=j) & (dist[i][j]<=thresh)):
        p1.append(i)
        p2.append(j)
        d.append(dist[i][j])
  return p1,p2,d
def change_2_red(img,person,p1,p2):
  risky = np.unique(p1+p2)
  points = []
  for i in risky:
    x1,y1,x2,y2 = person[i]
    _ = cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
    points.append((int((x1+x2)/2), int(y2)))
  return img, points
################################ Video/image processing functions ################################
def video_to_frames(video_path, num_frames, temp_dir="frames/"):
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    ret, first_frame = cap.read()
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # save each frame to folder
            img = temp_dir + str(cnt) + '.png'
            cv2.imwrite(img, frame)
            cnt = cnt + 1
            if (cnt == num_frames):
                break
        # Break the loop
        else:
            break
    return cnt
            
def load_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data

def detect_with_yolo(out = 'yolo_output/', source = 'frames/'):
    weights = ['weights/yolov5l.pt']
    save_txt = True
    view_img = False
    save_txt = True
    imgsz = 640
    conf_thres = 0.4
    iou_thres = 0.5
    classes = []
    save_img = True
    with torch.no_grad():
        if True:
            detect(out, source, weights, view_img, save_txt, imgsz)
def generate_video(image_path, num=300):
    # image_path should be where the images are stored, images must have the format of
    # i.png with i being a number, image_path must end with a / for easy of string
    # operations
    imgs = []
    img_name_template = image_path + "{}.png"
    for i in range(num):
        imgs.append(imageio.imread(img_name_template.format(i)))
    imageio.mimsave(image_path + "0movie.gif", imgs)

#----------------------------------- Birds Eye View -------------------------------------------
def read_dict():
    with open('yolo_output/json_out.txt', 'r') as inf:
        dict = eval(inf.read())
        return dict

def helper(dict, matrix, imgOutput, thresh=100):
    for frame in dict:
        boxes = dict[frame]["location_x"]
        scores = dict[frame]["confidence"]
        classes = dict[frame]["category"]
        img = cv2.imread(frame)
        height, width, _ = img.shape

        array_boxes_detected = get_human_box_detection(boxes, scores, classes, height, width)   # all human boxes in 1 frame
        print(array_boxes_detected)

        img = cv2.imread(frame)
        copy = img.copy()
        draw(copy, array_boxes_detected[0])
        draw(copy, array_boxes_detected[1])
        plt.imshow(copy)
        plt.show()

        array_centroids, array_groundpoints = get_centroids_and_groundpoints(array_boxes_detected)  # 1 gound point for each box
        print(array_centroids)
        print(array_groundpoints)

        transformed_midpoints = compute_point_perspective_transformation(matrix, array_groundpoints) # transformed bottom centre points
        print(transformed_midpoints)

        cv2.circle(imgOutput, (int(transformed_midpoints[0][0]), int(transformed_midpoints[0][1])), 20, (0, 0, 255), -1)
        cv2.circle(imgOutput, (int(transformed_midpoints[1][0]), int(transformed_midpoints[1][1])), 20, (0, 255, 0), -1)
        plt.imshow(imgOutput)
        plt.show()

        dist = compute_distance(transformed_midpoints, len(transformed_midpoints))      # computing euclidean between every pair of centre points
        p1, p2, d = find_closest(dist, len(transformed_midpoints), thresh)
        img, points = change_2_red(img, array_boxes_detected, p1, p2)

        break
    return

def change_2_red(img,array_boxes_detected,p1,p2):
  risky = np.unique(p1+p2)
  points = []
  for i in risky:
    x1,y1,x2,y2 = array_boxes_detected[i]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,255), 5)
    points.append((int((x1+x2)/2), int(y2)))
  plt.imshow(img)
  plt.show()
  return img, points

def compute_distance(midpoints,num):
  dist = np.zeros((num,num))
  for i in range(num):
    for j in range(i+1,num):
      if i!=j:
        dst = distance.euclidean(midpoints[i], midpoints[j])
        dist[i][j]=dst
  return dist

def find_closest(dist,num,thresh):
  p1=[]
  p2=[]
  d=[]
  for i in range(num):
    for j in range(i,num):
      if( (i!=j) & (dist[i][j]<=thresh)):
        p1.append(i)
        p2.append(j)
        d.append(dist[i][j])
  return p1,p2,d

list_points = list()

def find_matrix():
     img_path = "/Users/victorzhang/Desktop/EyeDK/frames/1.png"
     img = cv2.imread(img_path)
     width, height, _ = img.shape
     windowName = 'MouseCallback'
     cv2.namedWindow(windowName)
     cv2.setMouseCallback(windowName, CallBackFunc)

     while (True):
         cv2.imshow(windowName, img)
         if len(list_points) == 4:
             # Return a dict to the YAML file
             config_data = dict(
                 image_parameters=dict(
                     p2=list_points[3],
                     p1=list_points[2],
                     p4=list_points[0],
                     p3=list_points[1],
                     width_og=width,
                     height_og=height,
                     img_path=img_path,
                     size_frame=width,
                 ))
             # Write the result to the config file
             with open('birdview.yml', 'w') as outfile:
                 yaml.dump(config_data, outfile, default_flow_style=False)
             break
         if cv2.waitKey(20) == 27:
             break
     cv2.destroyAllWindows()

     with open("birdview.yml", "r") as ymlfile:
         cfg = yaml.safe_load(ymlfile)
     width_og, height_og = 0, 0
     corner_points = []
     for section in cfg:
         corner_points.append(cfg["image_parameters"]["p1"])
         corner_points.append(cfg["image_parameters"]["p2"])
         corner_points.append(cfg["image_parameters"]["p3"])
         corner_points.append(cfg["image_parameters"]["p4"])
         width_og = int(cfg["image_parameters"]["width_og"])
         height_og = int(cfg["image_parameters"]["height_og"])
         img_path = cfg["image_parameters"]["img_path"]
         size_frame = cfg["image_parameters"]["size_frame"]

     matrix, imgOutput = compute_perspective_transform(corner_points, width_og, height_og, img)
     return matrix, imgOutput

def compute_perspective_transform(corner_points, width, height, image):
    """
     Compute the transformation matrix
    @ corner_points : 4 corner points selected from the image
    @ height, width : size of the image
    return : transformation matrix and the transformed image
    """
    corner_points_array = np.float32(corner_points)
    img_params = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(corner_points_array, img_params)
    img_transformed = cv2.warpPerspective(image, matrix, (width, height))
    return matrix, img_transformed

def CallBackFunc(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left button of the mouse is clicked - position (", x, ", ", y, ")")
        list_points.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("Right button of the mouse is clicked - position (", x, ", ", y, ")")
        list_points.append([x, y])

def compute_point_perspective_transformation(matrix, list_downoids):
    """ Apply the perspective transformation to every ground point which have been detected on the main frame.

	@ matrix : the 3x3 matrix 
	@ list_downoids : list that contains the points to transform
	return : list containing all the new points
	"""
    # Compute the new coordinates of our points
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    # Loop over the points and add them to the list that will be returned
    transformed_points_list = list()
    for i in range(0, transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0], transformed_points[i][0][1]])
    return transformed_points_list

def get_human_box_detection(boxes,scores,classes,width,height):
    """ 
	For each object detected, check if it is a human and if the confidence >> our threshold.
	Return 2 coordonates necessary to build the box.
	@ boxes : all our boxes coordinates
	@ scores : confidence score on how good the prediction is -> between 0 & 1
	@ classes : the class of the detected object ( 1 for human )
	@ height : of the image -> to get the real pixel value
	@ width : of the image -> to get the real pixel value
	"""
    array_boxes = list() # Create an empty list
    for i in range(len(boxes)):
        # If the class of the detected object is person and the confidence of the prediction is > 0.8
        if classes[i] == 0.0 and scores[i] > 0.80:
            # Multiply the X coordonnate by the height of the image and the Y coordonate by the width
		    # To transform the box value into pixel coordonate values.
            box = [boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]] * np.array([height, width, height, width])
            # Add the results converted to int
            array_boxes.append([int(box[0]),int(box[1]),int(box[2]),int(box[3])])
    return array_boxes

def get_centroids_and_groundpoints(array_boxes_detected):
    """
	For every bounding box, compute the centroid and the point located on the bottom center of the box
	@ array_boxes_detected : list containing all our bounding boxes 
	"""
    array_centroids, array_groundpoints = [],[] # Initialize empty centroid and ground point lists
    for index, box in enumerate(array_boxes_detected):
        centroid, ground_point = get_points_from_box(box)
        array_centroids.append(centroid)
        array_groundpoints.append(ground_point)
    return array_centroids, array_groundpoints

def get_points_from_box(box):
    """
	Get the center of the bounding and the point "on the ground"
	@ param = box : 2 points representing the bounding box
	@ return = centroid (x1,y1) and ground point (x2,y2)
	"""
    # Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
    center_x = int(((box[0]+box[2])/2)) #top left -> bottom right corner (bottom right has higher values)
    center_y = int(((box[1]+box[3])/2))
    center_y_ground = center_y + ((box[2] - box[0])/2) # box[2] or box[1] - whichever is bottom
    return (center_x, center_y), (center_x, int(box[3]))  # prev was int(center_y_ground), box[0]

def draw(img, corners):
    cv2.circle(img, (corners[0], corners[1]), 40, (0, 0, 255), 2)
    cv2.circle(img, (corners[2], corners[3]), 40, (0, 255, 0), 2)

if __name__ == "__main__":
    # detect_with_yolo()
    # generate_video("yolo_output/")

    json_dict = read_dict()
    #print(dict["/Users/victorzhang/Desktop/EyeDK/frames/102.png"]["location_x"][0])

    matrix, imgOutput = find_matrix()
    plt.imshow(imgOutput)
    plt.show()

    helper(json_dict, matrix, imgOutput)