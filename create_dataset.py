import os
import numpy as np
import cv2
import Augmentor
import shutil

# Move images from original dataset to target directories of train / test / val
def moveImagesFromOriginalDataset():
    print(os.getcwdb())
    print("Starting to move images from original dataset into yolo target directories")
    print("Starting with train images")
    source_dir_2012 = './original_dataset/Ara2012'
    source_dir_2013 = './original_dataset/Ara2013-Canon'

    # Train files - 90 files from 2012, 120 from 2013
    list_ara2012_train = ["%03d" % i for i in range(1, 91)]
    list_ara2013_canon_train = ["%03d" % i for i in range(1, 121)]
    target_dir_rgb = './yolov3/leaf_data/images/train'
    target_dir_label = './yolov3/leaf_data/labels/train'
    for file_index in list_ara2012_train:
        filepath_rgb = source_dir_2012 + "/ara2012_plant" + file_index + "_rgb.png"
        shutil.copy(filepath_rgb, target_dir_rgb)
        rename_path = target_dir_rgb + "/ara2012_plant" + file_index
        os.rename(rename_path + "_rgb.png", rename_path + ".png")

        filepath_label = source_dir_2012 + "/ara2012_plant" + file_index + "_label.png"
        shutil.copy(filepath_label, target_dir_label)
        rename_path = target_dir_label + "/ara2012_plant" + file_index
        os.rename(rename_path + "_label.png", rename_path + ".png")

    for file_index in list_ara2013_canon_train:
        filepath_rgb = source_dir_2013 + "/ara2013_plant" + file_index + "_rgb.png"
        shutil.copy(filepath_rgb, target_dir_rgb)
        rename_path = target_dir_rgb + "/ara2013_plant" + file_index
        os.rename(rename_path + "_rgb.png", rename_path + ".png")

        filepath_label = source_dir_2013 + "/ara2013_plant" + file_index + "_label.png"
        shutil.copy(filepath_label, target_dir_label)
        rename_path = target_dir_label + "/ara2013_plant" + file_index
        os.rename(rename_path + "_label.png", rename_path + ".png")

        # test files - 90-110 from 2012, 120-150 from 2013
    print("Starting with test images")
    list_ara2012_test = ["%03d" % i for i in range(91, 111)]
    list_ara2013_canon_test = ["%03d" % i for i in range(121, 151)]
    target_dir_rgb = './yolov3/leaf_data/images/test'
    target_dir_label = './yolov3/leaf_data/labels/test'
    for file_index in list_ara2012_test:
        filepath_rgb = source_dir_2012 + "/ara2012_plant" + file_index + "_rgb.png"
        shutil.copy(filepath_rgb, target_dir_rgb)
        rename_path = target_dir_rgb + "/ara2012_plant" + file_index
        os.rename(rename_path + "_rgb.png", rename_path + ".png")

        filepath_label = source_dir_2012 + "/ara2012_plant" + file_index + "_label.png"
        shutil.copy(filepath_label, target_dir_label)
        rename_path = target_dir_label + "/ara2012_plant" + file_index
        os.rename(rename_path + "_label.png", rename_path + ".png")

    for file_index in list_ara2013_canon_test:
        filepath_rgb = source_dir_2013 + "/ara2013_plant" + file_index + "_rgb.png"
        shutil.copy(filepath_rgb, target_dir_rgb)
        rename_path = target_dir_rgb + "/ara2013_plant" + file_index
        os.rename(rename_path + "_rgb.png", rename_path + ".png")

        filepath_label = source_dir_2013 + "/ara2013_plant" + file_index + "_label.png"
        shutil.copy(filepath_label, target_dir_label)
        rename_path = target_dir_label + "/ara2013_plant" + file_index
        os.rename(rename_path + "_label.png", rename_path + ".png")

    # validate files
    print("Starting with validation images")
    list_ara2012_val = ["%03d" % i for i in range(111, 121)]
    list_ara2013_canon_val = ["%03d" % i for i in range(151, 166)]
    target_dir_rgb = './yolov3/leaf_data/images/validate'
    target_dir_label = './yolov3/leaf_data/labels/validate'
    for file_index in list_ara2012_val:
        filepath_rgb = source_dir_2012 + "/ara2012_plant" + file_index + "_rgb.png"
        shutil.copy(filepath_rgb, target_dir_rgb)
        rename_path = target_dir_rgb + "/ara2012_plant" + file_index
        os.rename(rename_path + "_rgb.png", rename_path + ".png")

        filepath_label = source_dir_2012 + "/ara2012_plant" + file_index + "_label.png"
        shutil.copy(filepath_label, target_dir_label)
        rename_path = target_dir_label + "/ara2012_plant" + file_index
        os.rename(rename_path + "_label.png", rename_path + ".png")

    for file_index in list_ara2013_canon_val:
        filepath_rgb = source_dir_2013 + "/ara2013_plant" + file_index + "_rgb.png"
        shutil.copy(filepath_rgb, target_dir_rgb)
        rename_path = target_dir_rgb + "/ara2013_plant" + file_index
        os.rename(rename_path + "_rgb.png", rename_path + ".png")

        filepath_label = source_dir_2013 + "/ara2013_plant" + file_index + "_label.png"
        shutil.copy(filepath_label, target_dir_label)
        rename_path = target_dir_label + "/ara2013_plant" + file_index
        os.rename(rename_path + "_label.png", rename_path + ".png")

    print("Finished moving images")

# Create set of augmented images
def createAugmentedDataset():
    # Augment train:
    images_data_directory = './yolov3/leaf_data/images/train/'
    label_images_data_directory = './yolov3/leaf_data/labels/train/'
    num_images_to_generate = 5
    augment(images_data_directory, label_images_data_directory, num_images_to_generate)

    # Augment test:
    images_data_directory = './yolov3/leaf_data/images/test/'
    label_images_data_directory = './yolov3/leaf_data/labels/test/'
    num_images_to_generate = 20
    augment(images_data_directory, label_images_data_directory, num_images_to_generate)

    # Augment val:
    images_data_directory = './yolov3/leaf_data/images/validate/'
    label_images_data_directory = './yolov3/leaf_data/labels/validate/'
    num_images_to_generate = 5
    augment(images_data_directory, label_images_data_directory, num_images_to_generate)

# create augmented images from specific locations, for train / test / val
def augment(images_path, labels_path, num_to_gen):
    p = Augmentor.Pipeline(images_path)
    # Point to a directory containing ground truth original_yolo_data.
    # Images with the same file names will be added as ground truth original_yolo_data
    # and augmented in parallel to the original original_yolo_data.
    p.ground_truth(labels_path)

    # Add operations to the pipeline
    p.rotate_random_90(probability=0.4)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.flip_top_bottom(probability=0.5)

    # Make me images bitch!
    p.sample(1)

# Move augmented images to the right dir
def move_Agumented():
    pass


def createLabels(labels_src):
    index = 0
    src_files = os.listdir(labels_src)
    total = len(src_files)
    for file in src_files:
        print(f"At file: {file}, file {index} out of {total}")
        segmented_image = cv2.imread(labels_src+file)
        f = open((labels_src + file)[:-4] + ".txt", "w+")
        image_shape = np.shape(segmented_image)
        image_height = image_shape[0]
        image_width = image_shape[1]
        list_of_colors = np.unique(
            np.reshape([color for color in segmented_image[:, :]], newshape=(image_shape[0] * image_shape[1], 3)),
            axis=0)

        for color in list_of_colors[1:]:
            # Get the masked pixels (all the pixels of this color)
            [leaf_y, leaf_x] = np.where(np.all(segmented_image == color, axis=-1))
            # Build ze bounding box
            bbox = [np.min(leaf_x), np.min(leaf_y), np.max(leaf_x), np.max(leaf_y)]
            box_width = min((bbox[2] - bbox[0]) / image_width, 1)  # Width of bbox, max 1
            box_height = min((bbox[3] - bbox[1]) / image_height, 1)  # Height of bbox
            normalized_bbox = [min(bbox[0] / image_width, 1), min(bbox[1] / image_height, 1),
                               # Normalize the important 2
                               min(bbox[2] / image_width, 1), min(bbox[3] / image_height, 1)]  # And also the other 2
            # Write bounding box to file
            f.write("%d %.6f %.6f %.6f %.6f\n" % (0, normalized_bbox[0], normalized_bbox[1], box_width, box_height))

        index += 1

# Create labels
def setupCreateLabels():
    print("Starting with label-creation")
    # Generates BBOXs based on the segmented file:
    # For the original images
    src = "./yolov3/leaf_data/labels/train/"
    createLabels(src)
    src = "./yolov3/leaf_data/labels/test/"
    createLabels(src)
    src = "./yolov3/leaf_data/labels/validate/"
    createLabels(src)

    # For the augmented images
    src = "./yolov3/leaf_data/labels/train/output/"
    createLabels(src)
    src = "./yolov3/leaf_data/labels/test/output/"
    createLabels(src)
    src = "./yolov3/leaf_data/labels/validate/output/"
    createLabels(src)

    print("Finished generating labels")

# Create txt files containing train / test / val file names
def prepareTxts():
    # Create the train.txt and text.txt
    directory = "./yolov3/leaf_data/labels/train/"
    target_file_train = "./yolov3/leaf_data/leaf_train.txt"
    target_file_test = "./yolov3/leaf_data/leaf_test.txt"
    target_file_val = "./yolov3/leaf_data/leaf_val.txt"
    shapes_target_file_train = "./yolov3/leaf_data/leaf_train.shapes"
    shapes_target_file_test = "./yolov3/leaf_data/leaf_test.shapes"
    shapes_target_file_val = "./yolov3/leaf_data/leaf_.shapes"
    all_targets = [[target_file_train, shapes_target_file_train], [target_file_test, shapes_target_file_test],
                   [target_file_val, shapes_target_file_val]]
    for targets in all_targets:
        createTxts(directory, targets[0], targets[1])

    print("Finished generating the train.txt file and .shapes file")

def createTxts(images_directory, txt_target, shape_target):
    images_list_to_write = []
    shapes_to_write = []

    for file in os.listdir(images_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            images_list_to_write.append(images_directory + filename)
            image = cv2.imread(images_directory + filename)
            # Log width, height because im pretty sure that's what yolo expect
            shapes_to_write.append([image.shape[1], image.shape[0]])

    # Save txt
    np.savetxt(txt_target, images_list_to_write, delimiter="", newline="\n",
               fmt="%s")
    # Save .shapes
    np.savetxt(shape_target, shapes_to_write, delimiter=" ", newline="\n",
               fmt="%s %s")

# Create yaml

if __name__ == "__main__":
    moveImagesFromOriginalDataset()
    createAugmentedDataset()
    createLabels()
    prepareTxts()