import cv2
from scipy.spatial import distance
import numpy as np
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
def video_to_frames(video_path, num_frames):
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
            img = 'frames/' + str(cnt) + '.png'
            cv2.imwrite(img, frame)
            cnt = cnt + 1
            if (cnt == num_frames):
                break
        # Break the loop
        else:
            break