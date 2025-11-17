from numpy.typing import NDArray
from typing import Dict, List
from ultralytics import YOLO
import cv2


class ZoneDetection:
    def __init__(self):
        # Load model from pt
        self.model = YOLO("models/yolo11n_D-Fire.pt")

    def get_zones(self, img: NDArray) -> List[Dict]:
        """
        Params:
        img: NDArray 
            Input image as a numpy array.

        Returns:
        List[Dict]
            A list of detected zones/objects. Each item is represented as a dictionary with keys:
            - 'type': str, the type of the detected item (e.g., 'fire', 'smoke').
            - 'bbox': Tuple[float], bounding box coordinates [x1, y1, x2, y2].
            - 'score': float, confidence score of the detection. (0-1 range)
        """
        # Run inference
        results = self.model(img)
        #results = self.model(img, save=True) # if we wanna see the result image with the detected boxes

        # Parse results
        parsed_results = []
        r = results[0] # YOLO returns a list → we take the first (and only) result
        for box in r.boxes: # Iterate over detected boxes
            cls_id = int(box.cls[0])           # numeric class ID
            cls_name = r.names[cls_id]         # class name ("fire"/"smoke")
            conf = float(box.conf[0])          # confidence score
            xyxy = box.xyxy[0].tolist()        # [x1, y1, x2, y2]

            parsed_results.append({
                "type": cls_name,
                "bbox": [float(x) for x in xyxy],
                "score": conf
            })
        return parsed_results


image = cv2.imread("D-Fire_dataset/test/images/WEB10490.jpg")
zone_detection = ZoneDetection()
zone_detection.get_zones(image)