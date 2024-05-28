from ultralytics import YOLO
import glob
import os
import numpy as np
import cv2

model = YOLO('yolov8n-face.pt')
with open("SD1.5_bad.txt", "w") as f:
    images = glob.glob("../data/TrueFaceSD/TrueFace_PreSocial/Fake/SD_1.5_simple/*.png")
    split = np.array_split(np.asarray(images), np.rint(len(images)/800))
    for index, image in enumerate(split):
        print(index, "/", len(split))
        # Perform object detection on an image using the model
        results = model.predict(list(image), verbose=False)
        for i, result in enumerate(results):
            #print(dir(result))
            if len(result.boxes.cls) != 1:
                f.write(result.path+'\n')
                #print(result.path, len(result.boxes.cls))
