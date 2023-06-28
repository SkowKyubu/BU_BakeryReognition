import torch
from ultralytics import YOLO
import cv2
import numpy as np
import math
import cvzone
# Charger les poids entrainés
#model = YOLO("best2.pt")

# Définir les noms de classe
classNames = ["cannele","cookie","croissant","donuts","pain_chocolat"]


def load_model():
    model = YOLO("best2.pt")
    return model

def price(cakes):
    amount = 0
    for cake in cakes:
        if cake == 0:
            amount = amount + 100  # cannele
        elif cake == 1:
            amount = amount + 103  # cookie
        elif cake == 2:
            amount = amount + 101  # croissant
        elif cake == 3:
            amount = amount + 108  # donuts
        elif cake == 4:
            amount = amount + 109  # pain au chocolat
    return amount

def detect(img,model,print_detection):
    results = model(img, stream=True)
    cakes = []
    for r in results:  # On trace le rectangle autour de chaque objet détecté
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100  # proba de prediction
            # class Name
            cls = int(box.cls[0])
            cakes.append(cls)
            if print_detection:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (x1,y1),(x2,y2),(255,0,255),2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # prediction/ confidence
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    #text = print("You have to pay " + str(price(cakes)) + " Baths")
    return price(cakes),img


#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("image/traffic.mp4")

"""
# AVEC IMAGE
cap = cv2.imread ("C:/Users/artga/Pictures/Camera Roll/WIN_20230616_11_10_16_Pro.jpg")
img = cap.copy()  # Process the imported image directly
results = model(img, stream=True)
cakes = []
for r in results: # On trace le rectangle autour de chaque objet détecté
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1,y1),(x2,y2),(255,0,255),2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # prediction/ confidence
            conf = math.ceil((box.conf[0] * 100)) / 100  # proba de prediction
            # class Name
            cls = int(box.cls[0])
            cakes.append(cls)

            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

print(cakes)
print("You have to pay " + str(price(cakes)) + " Baths")
while True:
    cv2.imshow("Image", img)
    if cv2.waitKey(0) & 0xFF == ord('q'):  # Wait for a key press and check if it's 'q' to quit
        break

cv2.destroyAllWindows()


import subprocess

command = "yolo task=detect mode=predict model=best2.pt conf=0.25 source=test/img.jpg"
subprocess.call(command, shell=True)
"""