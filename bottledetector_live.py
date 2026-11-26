##################################################
## Botell.ai
##
## Software package that detects and counts bottles in streams.
##
##################################################
## Author: Andrew Heller and Jason Davison
## Copyright: Copyright 2024, Botell.ai
## License: Creative Commons Attribution 4.0 International
## Version: 1.2.5
## Maintainer: Andrew Heller
## Email: abh037@gmail.com and davisonj@cua.edu
## Status: In-progress -- 8/14/2024 last update
##################################################


from ultralytics import YOLO
from ultralytics.engine.results import Results
from norfair import Detection
from typing import List
import numpy as np
import cv2


class BottleDetector:

    def __init__(self, weights='bottle_weights.pt'):
        """
        Description:
            initializes a new YOLO model
        Parameters:
            weights (String)    : path to the .pt file containing the weights of the model
        Returns:
            BottleDetector class object
        """
        self.model = YOLO(weights)
            
            
    def predict_frame(self, frame, thresh):
        """
        Description:
            runs the input frame through an inference pass on the class's model
        Parameters:
            frame (String)      : video frame to be processed and tracked
            slice (bool)        : whether the frame should be sliced via sahi for
                                  processing (note that this increases accuracy for
                                  detecting small objects, but increases inference time)
        Returns:
            List[Detections]: list of norfair Detection objects for the results of
                              running the model on the input frame
        """
        result = self.model(frame, conf=thresh, verbose=False, iou=0.7)
        result, pos = self._yolo_detections_to_norfair_detections(result)
        return result, pos
        
        
    #adapted from https://github.com/tryolabs/norfair/blob/master/demos/yolov7/src/demo.py
    def _yolo_detections_to_norfair_detections(self, yolo_detections: Results, track_points = "centroid") -> List[Detection]:
        """convert detections_as_xywh to norfair detections"""
        norfair_detections: List[Detection] = []
        boxes = yolo_detections[0].boxes.cpu()
        bbox = []

        for i, _ in enumerate(boxes.cls):

            bbox = np.array(
                [
                    [boxes.xyxy[i][0], boxes.xyxy[i][1]],
                    [boxes.xyxy[i][2], boxes.xyxy[i][3]],
                ]
            )
            scores = np.array(
                [boxes.conf[i], boxes.conf[i]]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=0
                )
            )  
        return norfair_detections, bbox


if __name__ == "__main__":


    cap = cv2.VideoCapture(0)
    model = BottleDetector()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow('Live Video', frame)
        nf_detections, pos = model.predict_frame(frame, 0.5)
        print(pos)
        print(len(nf_detections))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


