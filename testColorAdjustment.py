import cv2
from operator import itemgetter
import numpy as np
from yolo_segmentation import YOLOSegmentation

cap = cv2.VideoCapture("vid1.mov")

ys = YOLOSegmentation("yolov8m-seg.pt")

# Define the groups and corresponding colors
groups = {
    "Team A": (0, 255, 0),   # Green color for Team A
    "Team B": (255, 0, 0),   # Red color for Team B
    "Referee": (0, 0, 255)   # Blue color for the referee
}


def get_average_color(a):
    avg_color_per_row = np.average(a, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color

# Function to automatically generate color mappings
def generate_color_map(groups):
    color_map = {}
    # Additional colors if needed
    colors = iter([(0, 0, 0), (128, 128, 128), (192, 192, 192)])
    for group in groups:
        color_map[group] = next(colors)
    return color_map


# Generate color mappings for the groups
color_map = generate_color_map(groups)


def perspective_transform(p):
    tl = [915, 254]
    bl = [160, 520]
    tr = [1692, 288]
    br = [1088, 567]

    tl2 = [72, 61]
    bl2 = [72, 181]
    tr2 = [130, 61]
    br2 = [130, 181]

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([tl2, bl2, tr2, br2])

    pts3 = np.float32([p])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    pts3o = cv2.perspectiveTransform(pts3[None, :, :], M)
    x = int(pts3o[0][0][0])
    y = int(pts3o[0][0][1])
    new_p = (x, y)
    return new_p


while True:
    ret, frame = cap.read()
    dst = cv2.imread("dst.jpg")
    if not ret:
        break

    frame2 = np.array(frame)

    bboxes, classes, segmentations, scores = ys.detect(frame)

    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        if class_id in color_map:
            (x, y, x2, y2) = bbox

            minX = min(seg, key=itemgetter(0))[0]
            maxY = max(seg, key=itemgetter(1))[1]

            bottomVal = int(2 * (maxY - seg[0][1]) / 3 + seg[0][1])

            roi = frame2[seg[0][1]:bottomVal, seg[0][0]:seg[len(seg)-1][0]]

            dominant_color = get_average_color(roi)

            point = perspective_transform([minX, maxY])

            group = "Referee"  # Default group is "Referee"

            # Determine the group based on class ID or any other condition
            if class_id == 0:
                group = "Team A"
            elif class_id == 1:
                group = "Team B"

            color = color_map[group]  # Get color based on the group

            cv2.polylines(frame, [seg], True, color, 2)
            cv2.circle(frame, (minX, maxY), 5, color, -1)

            cv2.circle(dst, point, 5, color, -1)

    cv2.imshow("Img", frame)
    cv2.imshow("top", dst)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
