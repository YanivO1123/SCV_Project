import pandas as pd
import os
import numpy as np


# c1 = lower left corner
# c2 = upper left corner
# c3 = upper right corner
# c4 = lower right corner

list_ara2012 = ["%03d" % i for i in range(1, 121)]
list_ara2013_canon = ["%03d" % i for i in range(1, 166)]

# for i in list_ara2012:
#     print(i)
#
#     data = pd.read_csv("original_dataset/Ara2012/ara2012_plant" + i + "_bbox.csv")
#     data.columns =['index', 'c1x', 'c1y', 'c2x', 'c2y', 'c3x', 'c3y', 'c4x', 'c4y']
#
#     f = open("labels/Ara2012/ara2012_plant" + i + ".txt","w+")
#
#     for i in range(len(data.index)):
#         x_coordinate = data.iloc[i]['c2x']
#         y_coordinate = data.iloc[i]['c2y']
#         width = data.iloc[i]['c3x'] - data.iloc[i]['c2x']
#         height = data.iloc[i]['c2y'] - data.iloc[i]['c1y']
#         dataForYolo = [x_coordinate, y_coordinate, width, height]
#         f.write("%d %d %d %d %d\n" % (1, dataForYolo[0], dataForYolo[1], dataForYolo[2], dataForYolo[3]))
#     f.close()


# Create the train.txt and text.txt
images_list_to_write = []
directory = os.fsencode("original_dataset/Ara2012/")
target_file = open("leaf_2012_train.txt", "w+")

# iterate over the files and add them to array
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png") and "rgb" in filename:
        images_list_to_write.append(filename)

# save the array into a txt file
np.savetxt(f"PyTorch_YOLOv4/leaf_data/{target_file.name}", images_list_to_write, delimiter="", newline="\n", fmt="%s")

