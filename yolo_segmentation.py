from ultralytics import YOLO
import numpy as np

class YOLOSegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    def detect(self, img):
        height, width, channels = img.shape
        
        results = self.model.predict(source=img.copy(), save=False, save_txt=False, device="mps")
        result = results[0]

        segmentation_contours_idx = []

        for seg in result.masks.segments:
            seg[:, 0] *= width
            seg[:, 1] *= height
            
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)
        
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype ="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

        return bboxes, class_ids, segmentation_contours_idx, scores