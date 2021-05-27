import Augmentor
import os
import numpy as np
import cv2

# Where is the data to augment?
images_data_directory = "all_images_location/actual_images_dir"
label_images_data_directory = "all_images_location/label_(color)_images_dir"

num_images_to_generate = 10000

# Instantiate the augmentor
p = Augmentor.Pipeline(images_data_directory)
# Point to a directory containing ground truth data.
# Images with the same file names will be added as ground truth data
# and augmented in parallel to the original data.
p.ground_truth(label_images_data_directory)

# Add operations to the pipeline
p.rotate_random_90(probability=0.4)
p.flip_left_right(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.flip_top_bottom(probability=0.5)

# Make me images bitch!
p.sample(num_images_to_generate)