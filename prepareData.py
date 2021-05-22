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
        # Get the masked pixels (all the pixels of this color)
        [leaf_y, leaf_x] = np.where(np.all(segmented_image == color, axis=-1))
        # Build ze bounding box
        bbox = [np.min(leaf_x), np.min(leaf_y), np.max(leaf_x), np.max(leaf_y)]
        box_width = min((bbox[2] - bbox[0]) / image_width, 1)  # Width of bbox, max 1
        box_height = min((bbox[3] - bbox[1]) / image_height, 1)  # Height of bbox
        normalized_bbox = [min(bbox[0] / image_width, 1), min(bbox[1] / image_height, 1),  # Normalize the important 2
                           min(bbox[2] / image_width, 1), min(bbox[3] / image_height, 1)]  # And also the other 2
        # Write bounding box to file
        f.write("%d %.6f %.6f %.6f %.6f\n" % (0, normalized_bbox[0], normalized_bbox[1], box_width, box_height))

print("Finished generating labels")

# Create the train.txt and text.txt
images_list_to_write = []
directory = os.fsencode("original_dataset/Ara2012/")
target_file_train = open("leaf_2012_train.txt", "w+")
target_file_test = open("leaf_2012_test.txt", "w+")
target_file_val = open("leaf_2012_val.txt", "w+")
shapes_target_file_train = "leaf_2012_train.shapes"
shapes_target_file_test = "leaf_2012_test.shapes"
shapes_target_file_val = "leaf_2012_val.shapes"

# iterate over the files and add them to array
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png") and "rgb" in filename:
        images_list_to_write.append("./images/"+filename)
# save the array into a txt file
np.savetxt(f"PyTorch_YOLOv4/leaf_data/{target_file_train.name}", images_list_to_write, delimiter="", newline="\n", fmt="%s")
np.savetxt(f"PyTorch_YOLOv4/leaf_data/{target_file_test.name}", images_list_to_write, delimiter="", newline="\n", fmt="%s")
np.savetxt(f"PyTorch_YOLOv4/leaf_data/{target_file_val.name}", images_list_to_write, delimiter="", newline="\n", fmt="%s")
np.savetxt(f"PyTorch_YOLOv4/leaf_data/{shapes_target_file_train}", shapes_to_write, delimiter=" ", newline="\n", fmt="%s %s")
np.savetxt(f"PyTorch_YOLOv4/leaf_data/{shapes_target_file_test}", shapes_to_write, delimiter=" ", newline="\n", fmt="%s %s")
np.savetxt(f"PyTorch_YOLOv4/leaf_data/{shapes_target_file_val}", shapes_to_write, delimiter=" ", newline="\n", fmt="%s %s")

print("Finished generating the train.txt file and .shapes file")