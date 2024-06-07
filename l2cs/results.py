from dataclasses import dataclass
import numpy as np
import json

@dataclass
class GazeResultContainer:

    pitch: np.ndarray
    yaw: np.ndarray
    bboxes: np.ndarray
    landmarks: np.ndarray
    scores: np.ndarray
    color: tuple

    def to_json(self):
        def convert_to_list(data):
            if isinstance(data, np.ndarray):
                return data.tolist()
            elif isinstance(data, list):
                return data
            else:
                return []

        data = {
            'pitch': convert_to_list(self.pitch),
            'yaw': convert_to_list(self.yaw),
            'bboxes': convert_to_list(self.bboxes),
            'landmarks': convert_to_list(self.landmarks),
            'scores': convert_to_list(self.scores),
            'color': convert_to_list(self.color)
        }
        return json.dumps(data)