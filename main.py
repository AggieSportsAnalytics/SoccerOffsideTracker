import cv2
from operator import itemgetter
import numpy as np
from yolo_segmentation import YOLOSegmentation

from functions import get_average_color, divide_colors

cap = cv2.VideoCapture("vid.mov")

ys = YOLOSegmentation("yolov8m-seg.pt")

font = cv2.FONT_HERSHEY_SIMPLEX

og_points = []
top_points = []

colors = []

player_coords = []

team1_bgr = [0, 0, 255]
team2_bgr = [255, 255, 255]

# Perspective transform function (pass in a point) (returns a point)
def perspective_transform(tl, bl, tr, br, tl2, bl2, tr2, br2):
    new_points = []
    for index, player in enumerate(player_coords):
        p = player

        pts1 = np.float32([tl,bl,tr,br])
        pts2 = np.float32([tl2,bl2,tr2,br2])

        pts3 = np.float32([p])

        M = cv2.getPerspectiveTransform(pts1,pts2)

        pts3o=cv2.perspectiveTransform(pts3[None, :, :], M)
        x = int(pts3o[0][0][0])
        y = int(pts3o[0][0][1])
        new_p = (x,y)

        new_points.append(new_p)

        # Place new point for each player
        if(index in group1_indices):
            cv2.circle(dst,new_p,5,team1_bgr,-1)
        if(index in group2_indices):
            cv2.circle(dst,new_p,5,team2_bgr,-1)

    # Find furthest player
    max_point_X, max_point_Y = min(new_points, key=itemgetter(0))[0], min(new_points, key=itemgetter(0))[1]
    cv2.putText(dst, "Last >", (max_point_X-55, max_point_Y+5), font, 0.5, (0,255,255), 1, cv2.LINE_AA)

    cv2.imshow('Top View', dst)

# Manage mouse clicks on original image
def og_click_event(event, x, y, _, __):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN and len(og_points) < 4:
        # Place marker on click point
        cv2.putText(frame, str(len(og_points)+1) ,(x, y-20), font, 1, (0, 255, 255), 2)
        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
        cv2.imshow('Img', frame)
        og_points.append([x, y])
        
        # Once all points are selected, call perspective transform
        if len(og_points) == 4 and len(top_points) == 4:
            perspective_transform(og_points[0], og_points[1], og_points[2], og_points[3], top_points[0], top_points[1], top_points[2], top_points[3])
            

# Manage mouse clicks on top view image
def top_click_event(event, x, y, _, __):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN and len(top_points) < 4:
        # Place marker on click point
        cv2.putText(dst, str(len(top_points)+1) ,(x, y-10), font, 0.5, (0, 255, 255), 1)
        cv2.circle(dst, (x, y), 5, (0, 255, 255), -1)
        cv2.imshow('Top View', dst)
        top_points.append([x, y])

        # Once all points are selected, call perspective transform
        if len(og_points) == 4 and len(top_points) == 4:
            perspective_transform(og_points[0], og_points[1], og_points[2], og_points[3], top_points[0], top_points[1], top_points[2], top_points[3])

# Loop through each frame
while True:
    # Video frame = frame
    ret, frame = cap.read()

    # 2D image = dst
    dst = cv2.imread("dst.jpg")

    if not ret:
        break

    # Copy of frame
    frame2 = np.array(frame)

    # Detect objects
    bboxes, classes, segmentations, scores = ys.detect(frame)

    player_coords.clear()
    colors.clear()

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

                newX = int((x2 - x)/3 + x)
                newY = int((y2 - y)/5 + y)
                newX2 = int(2*(x2 - x)/3 + x)
                newY2 = int(2*(y2 - y)/5 + y)

                # Shift based on player orientation
                if(distRight > distLeft):
                    # Shift left
                    newX = int(newX - ((distLeft + 30)/distRight)/5)
                    newX2 = int(newX2 - ((distLeft + 30)/distRight)/5)
                else:
                    # Shift right
                    newX = int(newX + ((distLeft + 30)/distRight)/5)
                    newX2 = int(newX2 + ((distLeft + 30)/distRight)/5)

                roi = frame2[newY:newY2, newX:newX2]

                # Get average color of smaller rectangle
                dominant_color = get_average_color(roi)

                colors.append(dominant_color)

                frame[y, x]

    group1_colors, group1_indices, group2_colors, group2_indices, group3_colors, group3_indices = divide_colors(colors, team1_bgr, team2_bgr)

    group1_avg = np.average(group1_colors, axis=0)
    group2_avg = np.average(group2_colors, axis=0)

    points = []

    # Loop through each object again
    for index, (bbox, class_id, seg, score) in enumerate(zip(bboxes, classes, segmentations, scores)):
         # If object is a player
         if class_id == 0:
            (x, y, x2, y2) = bbox


            if len(seg) != 0:
                minX = min(seg, key=itemgetter(0))[0]
                maxX = max(seg, key=itemgetter(0))[0]
                maxY = max(seg, key=itemgetter(1))[1]
            else:
                minX = x
                maxX = x2
                maxY = y2

            # If player is in group 1
            if(index in group1_indices):
                cv2.putText(frame, "Team 1", (x, y-5), font, 1, group1_avg, 3, cv2.LINE_AA)
                
                # Draw segmentation with the color of the dominant color of the player
                cv2.polylines(frame, [seg], True, team1_bgr, 3)
                cv2.circle(frame,(minX, maxY),5,team1_bgr,-1)
                player_coords.append([minX, maxY])

            # If player is in group 2
            if(index in group2_indices):
                cv2.putText(frame, "Team 2", (x, y-5), font, 1, group2_avg, 3, cv2.LINE_AA)

                # Draw segmentation with the color of the dominant color of the player
                cv2.polylines(frame, [seg], True, team2_bgr, 3)
                cv2.circle(frame,(minX, maxY),5,team2_bgr,-1)
                player_coords.append([minX, maxY])
    
    # Show images
    cv2.imshow("Img", frame)
    cv2.imshow("Top View", dst)

    cv2.setMouseCallback('Img', og_click_event)
    cv2.setMouseCallback('Top View', top_click_event)
    # Space to move forward a frame
    key = cv2.waitKey(0)
    # Esc to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()