import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

model = YOLO('../detect/train6/weights/best.pt')

# Confidence threshold
confidence_threshold = 0.7

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap = cv2.VideoCapture('000.mp4')

my_file = open("classes.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 6 != 0:
        continue
    frame = cv2.resize(frame, (512, 260))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        confidence = row[4]
        c = class_list[d]

        if confidence > confidence_threshold:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(frame, f'{c} {confidence:.2f}', (x1, y1), 1, 1)

    # Show confidence threshold
    cv2.putText(frame, f'Confidence Threshold: {confidence_threshold}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
