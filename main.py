import cv2
from operator import itemgetter
import numpy as np
from yolo_segmentation import YOLOSegmentation

from functions import get_average_color, classify_bgr_color

cap = cv2.VideoCapture("vid.mov")

ys = YOLOSegmentation("yolov8m-seg.pt")

font = cv2.FONT_HERSHEY_SIMPLEX

new_points = []

colors = []

player_coords = []

# INPUT TEAM COLORS (BGR)
team1_bgr = [143, 97, 164]
team2_bgr = [154, 115, 112]

# INPUT PERSPECTIVE COORDINATES ON ORIGINAL IMAGE (TL, BL, TR, BR)
og_perspective_coords = [[782, 349], [1585, 343], [96, 798], [1559, 801]]
# INPUT PERSPECTIVE COORDINATES ON NEW IMAGE (TL, BL, TR, BR)
new_perspective_coords = [[67, 31], [305, 32], [69, 652], [306, 652]]

# Perspective transform function (pass in a point) (returns a point)
def perspective_transform(player, team, original, new):
    tl, bl, tr, br = original
    tl2, bl2, tr2, br2 = new
    
    p = player

    pts1 = np.float32([tl,bl,tr,br])
    pts2 = np.float32([tl2,bl2,tr2,br2])

    pts3 = np.float32([p])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    pts3o=cv2.perspectiveTransform(pts3[None, :, :], M)
    x = int(pts3o[0][0][0])
    y = int(pts3o[0][0][1])
    new_p = (x,y)

    # Place transformed point for each player on dst
    if(team == "group1"):
        cv2.circle(dst,new_p,10,team1_bgr,-1)
        new_points.append(new_p)
    if(team == "group2"):
        cv2.circle(dst,new_p,10,team2_bgr,-1)
        new_points.append(new_p)

    cv2.imshow('Top View', dst)

# Loop through each frame
while True:
    # Video frame = frame
    ret, frame = cap.read()

    # 2D image = dst
    dst = cv2.imread("dst.png")

    if not ret:
        break

    # Copy of frame
    frame2 = np.array(frame)

    # Detect objects
    bboxes, classes, segmentations, scores = ys.detect(frame)

    player_coords.clear()
    colors.clear()
    new_points.clear()

    # Loop through each object
    for index, (bbox, class_id, seg, score) in enumerate(zip(bboxes, classes, segmentations, scores)):
        # If object is a player
        if class_id == 0:
            # Set corner coordinates for bounding box around player
            (x, y, x2, y2) = bbox
            
            # Draw segmentation around player
            if len(seg) != 0:
                minX = min(seg, key=itemgetter(0))[0]
                maxX = max(seg, key=itemgetter(0))[0]
                maxY = max(seg, key=itemgetter(1))[1]

                # Create smaller rectangle around player to use for color detection
                distLeft = int(abs(seg[0][0] - minX))
                distRight = int(abs(seg[0][0] - maxX))

                # Get smaller box points around player for detecting color
                newX = int((x2 - x)/3 + x)
                newY = int((y2 - y)/5 + y)
                newX2 = int(2*(x2 - x)/3 + x)
                newY2 = int(2*(y2 - y)/5 + y)

                # Shift color detection box based on player orientation
                if(distRight > distLeft):
                    # Shift left
                    newX = int(newX - ((distRight)/distLeft)/1.5)
                    newX2 = int(newX2 - ((distRight)/distLeft)/1.5)
                else:
                    # Shift right
                    newX = int(newX + ((distLeft)/distRight)*1.5)
                    newX2 = int(newX2 + ((distLeft)/distRight)*1.5)

                # Define smaller rectangle around player to use for color detection
                roi = frame2[newY:newY2, newX:newX2]

                # Get average color of smaller rectangle
                dominant_color = get_average_color(roi)
                cv2.rectangle(frame, (newX, newY), (newX2, newY2), dominant_color, 2)

                team = classify_bgr_color(dominant_color, team1_bgr, team2_bgr)

                if(team == "group1"):
                    cv2.putText(frame, "Team 1", (x, y-5), font, 1, team1_bgr, 3, cv2.LINE_AA)
                    
                    # Draw segmentation with the color of the dominant color of the player
                    cv2.polylines(frame, [seg], True, team1_bgr, 3)
                    cv2.circle(frame,(minX, maxY),5,team1_bgr,-1)
                if(team == "group2"):
                    cv2.putText(frame, "Team 2", (x, y-5), font, 1, team2_bgr, 3, cv2.LINE_AA)

                    # Draw segmentation with the color of the dominant color of the player
                    cv2.polylines(frame, [seg], True, team2_bgr, 3)
                    cv2.circle(frame,(minX, maxY),5,team2_bgr,-1)

        # Perspective transform for each player
        perspective_transform([minX, maxY], team, og_perspective_coords, new_perspective_coords)
    

    # Find furthest player and place vertical line
    max_point_X, max_point_Y = min(new_points, key=itemgetter(0))[0], min(new_points, key=itemgetter(0))[1]
    cv2.circle(dst, (max_point_X, max_point_Y), 10, (0,255,255), 2)
    cv2.line(dst, (max_point_X, 0), (max_point_X, 1035), (0,255,255), 2)

    # Show images
    cv2.imshow("Img", frame)
    cv2.imshow("Top View", dst)

    # Space to move forward a frame
    key = cv2.waitKey(0)
    # Esc to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()