#!/usr/local/bin/python3
import cv2
import pandas as pd

def plotbbox_old_labels():
    image_numbers = ["%03d" % i for i in range(1, 3)]

    for i in image_numbers:
        print(i)
        image = cv2.imread("original_dataset/Ara2012/ara2012_plant" + i + "_label.png")
        coordinates = pd.read_csv("leaf/labels/train/ara2012_plant" + i + "_rgb.txt", sep="\s+", header=None)
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

        # Save new image
        cv2.imwrite("leaf_data/show_bounding_boxes/bounding_boxes" + i + ".png", image)

def plotbbox_new_labels():
    image_numbers = ["%03d" % i for i in range(1, 4)]

    for i in image_numbers:
        print(i)
        image = cv2.imread("leaf_data/images/train/ara2012_plant" + i + ".png")
        coordinates = pd.read_csv("leaf_data/labels/train/ara2012_plant" + i + ".txt", sep="\s+", header=None)
        coordinates.columns = ['class', 'x_center', 'y_center', 'w', 'h']
        height = image.shape[0]
        width = image.shape[1]
        colors = [[0, 0, 255], [50, 68, 168], [50, 168, 58], [168, 164, 50], [224, 115, 20], [252, 3, 136],
                  [3, 252, 235], [0, 0, 255], [50, 68, 168], [50, 168, 58], [168, 164, 50], [224, 115, 20],
                  [252, 3, 136], [3, 252, 235], [0, 0, 255], [50, 68, 168], [50, 168, 58], [168, 164, 50],
                  [224, 115, 20], [252, 3, 136], [3, 252, 235], [0, 0, 255], [50, 68, 168], [50, 168, 58],
                  [168, 164, 50], [224, 115, 20], [252, 3, 136], [3, 252, 235]]
        for j in range(len(coordinates.index)):
            x_coordinate_center = round(coordinates.iloc[j]['x_center'] * width)
            y_coordinate_center = round(coordinates.iloc[j]['y_center'] * height)
            x_coordinate_left = x_coordinate_center - round(coordinates.iloc[j]['w'] * 0.5 * width)
            x_coordinate_right = x_coordinate_center + round(coordinates.iloc[j]['w'] * 0.5 * width)
            y_coordinate_top = y_coordinate_center - round(coordinates.iloc[j]['h'] * 0.5 * height)
            y_coordinate_bottom = y_coordinate_center + round(coordinates.iloc[j]['h'] * 0.5 * height)

            image = cv2.circle(image, (x_coordinate_center, y_coordinate_center), radius=3, color=(colors[j][0], colors[j][1], colors[j][2]), thickness=-1)
            image = cv2.circle(image, (x_coordinate_left, y_coordinate_top), radius=3, color=(colors[j][0], colors[j][1], colors[j][2]), thickness=-1)
            image = cv2.circle(image, (x_coordinate_right, y_coordinate_top), radius=3,color=(colors[j][0], colors[j][1], colors[j][2]), thickness=-1)
            image = cv2.circle(image, (x_coordinate_left, y_coordinate_bottom), radius=3,color=(colors[j][0], colors[j][1], colors[j][2]), thickness=-1)
            image = cv2.circle(image, (x_coordinate_right, y_coordinate_bottom), radius=3,color=(colors[j][0], colors[j][1], colors[j][2]), thickness=-1)

        # Save new image
        cv2.imwrite("leaf_data/show_bounding_boxes/bounding_boxes" + i + ".png", image)

if __name__ == "__main__":
    # plotbbox_old_labels()
    plotbbox_new_labels()