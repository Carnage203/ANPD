import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np


Zone_Polygon= np.array([
    [0,0],
    [0.5,0],
    [0.5,1],
    [0,1]
])

def parse_arguments() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Yolov8 live")
    parser.add_argument("--webcam-resolution",
    default=[1280,720],
    nargs=2,
    type=int)
    args=parser.parse_args()
    return args

def main():
    args=parse_arguments()
    frame_width,frame_height=args.webcam_resolution

    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

    model =YOLO('yolov8l.pt')

    box_annotator=sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon= (Zone_Polygon * np.array(args.webcam_resolution)).astype(int)
    zone=sv.PolygonZone(polygon=zone_polygon,frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator=sv.PolygonZoneAnnotator(zone=zone,
    color=sv.Color.red(),
    thickness=2,
    text_thickness=4,
    text_scale=2,
    text_color=sv.Color.white()
    )

    while True:
        ret, frame=cap.read()

        result=model(frame, agnostic_nms=True)[0]
        detections=sv.Detections.from_yolov8(result)
        '''use: class_id== [2,3,5,7]for car, motorcycle, bus and truck
        target_class_ids = [2, 3, 5, 7] specifies the class IDs you want to filter by.
        np.isin(detections.class_id, target_class_ids) creates a boolean array that is True where detections.class_id is one of the target class IDs.
        and remove NOT ~ from line 62 
        '''
        
        target_class_ids = [0]
        detections = detections[~np.isin(detections.class_id, target_class_ids)] 
         
        labels=[
            f'{model.model.names[class_id]} {confidence:0.2f}'
            for _, confidence, class_id,_ in detections
        ]
        frame=box_annotator.annotate(scene=frame, 
        detections=detections,
        labels=labels)

        zone.trigger(detections=detections)
        frame=zone_annotator.annotate(scene=frame)

        cv2.imshow("Yolov8",frame)

        if (cv2.waitKey(30) == 27):  #27-> ascii value for Esc
            break

if __name__=="__main__":
    main()