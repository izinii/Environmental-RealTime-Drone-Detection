from numpy.typing import NDArray
from typing import Dict, List


class ZoneDetection:
    def __init__(self):
        # Load model from pt
        self.model = ...

    def get_zones(self, img:NDArray) -> List[Dict]:
        """
        Params:
        img: NDArray
            Input image as a numpy array.

        Returns:
        List[Dict]
            A list of detected objects. Each object is represented as a dictionary with keys:
            - 'type': str, the type of the detected object (e.g., 'car', 'person').
            - 'bbox': Tuple[float], bounding box coordinates [x1, y1, x2, y2].
            - 'score': float, confidence score of the detection. (0-1 range)
        """
        # Run inference
        results = self.model(img)

        # Parse results
        parsed_results = ...
        return parsed_results