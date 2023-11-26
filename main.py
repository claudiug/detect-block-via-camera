from ultralytics import YOLO
import supervision as sv
import cv2
model = YOLO(model='yolov8n.pt')
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=2)
    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]
        frame_result = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(30) == 27:
            break
