import cv2
import numpy as np
from yolo_segmentation import YOLOSegmentation

cap = cv2.VideoCapture("vid.mov")

ys = YOLOSegmentation("yolov8m-seg.pt")


def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame2 = np.array(frame)

    bboxes, classes, segmentations, scores = ys.detect(frame)

    for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        if class_id == 0:
            (x, y, x2, y2) = bbox
            
            minY = np.max(seg[:, 1])
            bottomVal = int(2*(minY - seg[0][1])/3 + seg[0][1])
            
            a = frame2[seg[0][1]:bottomVal, seg[0][0]:seg[len(seg)-1][0]]

            cv2.polylines(frame, [seg], True, (0, 0, 225), 2)
            cv2.rectangle(frame, (seg[0][0], seg[0][1]), (seg[len(seg)-1][0], bottomVal), (225, 0, 0), 2)
            cv2.putText(frame, str(unique_count_app(a)), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (int(unique_count_app(a)[0]), int(unique_count_app(a)[1]), int(unique_count_app(a)[2])), 4)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()