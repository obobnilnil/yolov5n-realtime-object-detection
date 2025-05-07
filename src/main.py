# import cv2
# import time
# from ultralytics import YOLO

# model = YOLO("../yolov5su.pt")


# cap = cv2.VideoCapture(1)

# prev_time = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     start_time = time.time()

#     results = model.predict(frame, verbose=False)

#     annotated_frame = results[0].plot()

#     curr_time = time.time()
#     fps = 1 / (curr_time - prev_time)
#     prev_time = curr_time

#     cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     cv2.imshow("YOLOv5 Detection", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import time
from ultralytics import YOLO

# model = YOLO("../yolov5n.pt") Model loading from remote registry
model = YOLO("../yolov5nu.pt") # local model


cap = cv2.VideoCapture(1)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    results = model.predict(frame, imgsz=640, verbose=False)

    annotated_frame = results[0].plot()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLOv5n Detection (CPU)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


