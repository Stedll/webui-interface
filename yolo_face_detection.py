from ultralytics import YOLO
import glob
import os
import numpy as np
import cv2

model = YOLO('yolov8n-face.pt')
with open("SD1.5_bad.txt", "w") as f:
    images = glob.glob("data/SD1.5/*.png")
    for image in np.array_split(np.asarray(images), np.rint(len(images)/100)):
        # Perform object detection on an image using the model
        results = model.predict(list(image), verbose=False)
        for i, result in enumerate(results):
            #print(dir(result))
            if len(result.boxes.cls) != 1:
                f.write(result.path+'\n')
                #print(result.path, len(result.boxes.cls))
