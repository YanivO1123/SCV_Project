import pandas as pd
import os
import numpy as np
import cv2

list_ara2012 = ["%03d" % i for i in range(1, 121)]
list_ara2013_canon = ["%03d" % i for i in range(1, 166)]

# Find the shapes
directory = os.fsencode("original_dataset/Ara2012/")
shapes_to_write = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png") and "rgb" in filename:
        im = cv2.imread("original_dataset/Ara2012/"+filename)
        # print(im.shape[0], im.shape[1])
        shapes_to_write.append([im.shape[0], im.shape[1]])

print("Finished loading shapes")

# Generates BBOXs based on the segmented file:
for file_index in list_ara2012:
    print("At file number: ", file_index, " out of: ", len(list_ara2012))
    segmented_image = cv2.imread("original_dataset/Ara2012/ara2012_plant" + file_index + "_label.png")
    f = open("PyTorch_YOLOv4/leaf_data/labels/ara2012_plant" + file_index + "_rgb.txt", "w+")
    image_shape = np.shape(segmented_image)
    image_height = image_shape[0]
    image_width = image_shape[1]
    list_of_colors = np.unique(np.reshape([color for color in segmented_image[:,:]], newshape=(image_shape[0]*image_shape[1], 3)), axis=0)
    for color in list_of_colors[1:]:
        leaf_x = []
        leaf_y = []
        for x in range(len(segmented_image)):
            for y in range(len(segmented_image[x])):
                if color in segmented_image[x, y]:
                    # print(x, y)
                    leaf_x.append(x)
                    leaf_y.append(y)

        bbox = [np.min(leaf_x), np.min(leaf_y), np.max(leaf_x), np.max(leaf_y)]
        box_width = min((bbox[2] - bbox[0]) / image_width, 1)  # Width of bbox, max 1
        box_height = min((bbox[3] - bbox[1]) / image_height, 1)  # Height of bbox
        normalized_bbox = [min(bbox[0] / image_width, 1), min(bbox[1] / image_height, 1),      # Normalize the important 2
                           min(bbox[2] / image_width, 1), min(bbox[3] / image_height, 1)]  # And also the other 2

        f.write("%d %.6f %.6f %.6f %.6f\n" % (0, normalized_bbox[0], normalized_bbox[1], box_width, box_height))

# Old - generates BBOXes based on the BBOX data that appears wrong
# Genereate the labels, normalized
# index = 0
# counter = 0
# for file_index in list_ara2012:
#     data = pd.read_csv("original_dataset/Ara2012/ara2012_plant" + file_index + "_bbox.csv")
#     data.columns =['index', 'c1x', 'c1y', 'c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y']
#
#     f = open("PyTorch_YOLOv4/leaf_data/labels/ara2012_plant" + file_index + "_rgb.txt", "w+")
#
#     # if (index == 30):
#     #     print("Height of 31 should be: ", shapes_to_write[index][0], " and width: ", shapes_to_write[index][1])
#     #     for i in range(len(data.index)):
#     #         print("Point values are:", data.iloc[i])
#     for i in range(len(data.index)):
#         x_points = [data.iloc[i]['c1x'], data.iloc[i]['c2x'], data.iloc[i]['c3x'], data.iloc[i]['c4x']]
#         y_points = [data.iloc[i]['c1y'], data.iloc[i]['c2y'], data.iloc[i]['c3y'], data.iloc[i]['c4y']]
#
#         if np.min(x_points) < 0 or np.min(y_points) < 0:
#             print("Hey! Negative value in final Yolo data! at file: ", file_index, " and row: ", i)
#             print("x values: ", x_points)
#             print("y values: ", y_points)
#             counter += 1
#
#         x_coordinate = np.min(x_points)
#         y_coordinate = np.min(y_points)
#
#         width = (np.max(x_points) - x_coordinate) / shapes_to_write[index][1]
#         height = (np.max(y_points) - y_coordinate) / shapes_to_write[index][0]
#         dataForYolo = [x_coordinate / shapes_to_write[index][1], y_coordinate / shapes_to_write[index][0], width, height]
#
#         # Make sure that there are no > 1 values
#         for index_data in range(len(dataForYolo)):
#             if dataForYolo[index_data] > 1:
#                 dataForYolo[index_data] = 1
#                 print("Caught value larger than 1!")
#             # if dataForYolo[index_data] < 0:
#             #     print("Hey! Negative value in final Yolo data! at file: ", file_index, " and row: ", i)
#             #     print("And the values: ", dataForYolo)
#         f.write("%d %.6f %.6f %.6f %.6f\n" % (1, dataForYolo[0], dataForYolo[1], dataForYolo[2], dataForYolo[3]))
#
#         # if (dataForYolo[1] > 1):
#         #     print("index: ", index + 1, " and i: ", i)
#         #     print(dataForYolo[1])
#         #     print("Shape: ", shapes_to_write[index])
#         #     print("And y coordinate: ", y_coordinate)
#
#     f.close()
#     index += 1

print("Finished generating labels")

# Create the train.txt and text.txt
images_list_to_write = []
directory = os.fsencode("original_dataset/Ara2012/")
target_file = open("leaf_2012_train.txt", "w+")
shapes_target_file = "leaf_2012_train.shapes"

# iterate over the files and add them to array
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png") and "rgb" in filename:
        images_list_to_write.append("./images/"+filename)
        # print(filename)
        # im = cv2.imread("original_dataset/Ara2012/"+filename)
        # print(im.shape[0], im.shape[1])
        # shapes_to_write.append([im.shape[0], im.shape[1]])
# save the array into a txt file
np.savetxt(f"PyTorch_YOLOv4/leaf_data/{target_file.name}", images_list_to_write, delimiter="", newline="\n", fmt="%s")
np.savetxt(f"PyTorch_YOLOv4/leaf_data/{shapes_target_file}", shapes_to_write, delimiter=" ", newline="\n", fmt="%s %s")

print("Finished generating the train.txt file and .shapes file")