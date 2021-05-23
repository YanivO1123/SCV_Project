#!/usr/local/bin/python3
import cv2
import numpy as np
import pandas as pd

list_ara2012 = ["%03d" % i for i in range(119, 120)]
#list_ara2013_canon = ["%03d" % i for i in range(1, 166)]

for i in ["001"]:#list_ara2012:
    print(i)
    image = cv2.imread("original_dataset/Ara2012/ara2012_plant" + i + "_label.png")
    coordinates = pd.read_csv("PyTorch_YOLOv4/leaf_data/labels/ara2012_plant" + i + "_rgb.txt", sep="\s+", header=None)
    coordinates.columns =['class', 'x', 'y', 'w', 'h']
    height = image.shape[0]
    width = image.shape[1]
    colors = [[0, 0, 255], [50, 68, 168], [50, 168, 58], [168, 164, 50], [224, 115, 20], [252, 3, 136], [3, 252, 235], [0, 0, 255], [50, 68, 168], [50, 168, 58], [168, 164, 50], [224, 115, 20], [252, 3, 136], [3, 252, 235], [0, 0, 255], [50, 68, 168], [50, 168, 58], [168, 164, 50], [224, 115, 20], [252, 3, 136], [3, 252, 235], [0, 0, 255], [50, 68, 168], [50, 168, 58], [168, 164, 50], [224, 115, 20], [252, 3, 136], [3, 252, 235]]
    for j in range(len(coordinates.index)):
        x_coordinate_left = round(coordinates.iloc[j]['x'] * width)
        x_coordinate_right = round((coordinates.iloc[j]['x'] + coordinates.iloc[j]['w']) * width)
        y_coordinate_top = round(coordinates.iloc[j]['y'] * height)
        y_coordinate_bottom = round((coordinates.iloc[j]['y'] + coordinates.iloc[j]['h']) * height)

        image = cv2.circle(image, (x_coordinate_left, y_coordinate_top), radius=3, color=(colors[j][0], colors[j][1], colors[j][2]), thickness=-1)
        image = cv2.circle(image, (x_coordinate_right, y_coordinate_top), radius=3, color=(colors[j][0], colors[j][1], colors[j][2]), thickness=-1)
        image = cv2.circle(image, (x_coordinate_left, y_coordinate_bottom), radius=3, color=(colors[j][0], colors[j][1], colors[j][2]), thickness=-1)
        image = cv2.circle(image, (x_coordinate_right, y_coordinate_bottom), radius=3, color=(colors[j][0], colors[j][1], colors[j][2]), thickness=-1)

    #cv2.imwrite("images/Ara2012_with_bbox/ara2012_plant" + i + "_bbox.png", image)

# Save (just one for testing)
cv2.imwrite("images/Ara2012_with_bbox/result.png", image)


