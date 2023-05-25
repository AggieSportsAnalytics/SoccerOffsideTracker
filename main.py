import cv2
from operator import itemgetter
import numpy as np
from yolo_segmentation import YOLOSegmentation

cap = cv2.VideoCapture("vid.mov")

ys = YOLOSegmentation("yolov8m-seg.pt")

# Average color function (pass in box) (returns BGR)
def get_average_color(a):
    avg_color_per_row = np.average(a, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color

# Perspective transform function (pass in a point) (returns a point)
def perspective_transform(p):
    tl = [915,254]
    bl = [160,520]
    tr = [1692,288]
    br = [1088,567]

    tl2 = [72,61]
    bl2 = [72,181]
    tr2 = [130,61]
    br2= [130,181]

    pts1 = np.float32([tl,bl,tr,br])
    pts2 = np.float32([tl2,bl2,tr2,br2])

    pts3 = np.float32([p])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    pts3o=cv2.perspectiveTransform(pts3[None, :, :], M)
    x = int(pts3o[0][0][0])
    y = int(pts3o[0][0][1])
    new_p = (x,y)
    return new_p

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

    # Loop through each object
    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        # If object is a player
        if class_id == 0:
            # Set corner coordinates for bounding box around player
            (x, y, x2, y2) = bbox
            
            # Draw segmentation around player
            minX = min(seg, key=itemgetter(0))[0]
            maxY = max(seg, key=itemgetter(1))[1]

            # Create smaller rectangle around player to use for color detection
            bottomVal = int(2*(maxY - seg[0][1])/3 + seg[0][1])
            roi = frame2[seg[0][1]:bottomVal, seg[0][0]:seg[len(seg)-1][0]]

            # Get average color of smaller rectangle
            dominant_color = get_average_color(roi)

            # Get point to draw on 2D image based on the minimum X value (farthest right) and maximum Y value (lowest point)
            point = perspective_transform([minX, maxY])

            # Create random color
            for i in range(3):
                r = np.random.randint(0,255)
                b = np.random.randint(0,255)
                g = np.random.randint(0,255)

            # Draw segmentation with the color of the dominant color of the player
            cv2.polylines(frame, [seg], True, dominant_color, 3)
            # Draw point at the bottom right of the player
            cv2.circle(frame,(minX, maxY),5,dominant_color,-1)
            # Draw corresponding point on 2D image            
            cv2.circle(dst,point,5,dominant_color,-1)

    # Show images
    cv2.imshow("Img", frame)
    cv2.imshow("top", dst)

    # Space to move forward a frame
    key = cv2.waitKey(0)
    # Esc to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()