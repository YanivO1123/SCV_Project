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
    target_dir_rgb = 'leaf_data/images/train'
    target_dir_label = 'leaf_data/labels/train'
    for file_index in list_ara2012_train:
        # Copy the image
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
    target_dir_rgb = 'leaf_data/images/test'
    target_dir_label = 'leaf_data/labels/test'
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
    target_dir_rgb = 'leaf_data/images/validate'
    target_dir_label = 'leaf_data/labels/validate'
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
def createAugmentedDataset(number_auged_images_to_train, number_auged_images_to_test, number_auged_images_to_validate):
    # Augment train:
    images_data_directory = './leaf_data/images/train/'
    label_images_data_directory = './leaf_data/labels/train/'
    augment(images_data_directory, label_images_data_directory, number_auged_images_to_train)

    # Augment test:
    images_data_directory = './leaf_data/images/test/'
    label_images_data_directory = './leaf_data/labels/test/'
    augment(images_data_directory, label_images_data_directory, number_auged_images_to_test)

    # Augment val:
    images_data_directory = './leaf_data/images/validate/'
    label_images_data_directory = './leaf_data/labels/validate/'
    augment(images_data_directory, label_images_data_directory, number_auged_images_to_validate)

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
    p.sample(num_to_gen)

    # return p.get_ground_truth_paths()

# Move augmented images to the right dir
def move_Agumented():
    # Move train
    source_train = './leaf_data/images/train/output'
    target_train = './leaf_data/labels/train/output'
    move_files(source_train, target_train)

    # Move test
    source_test = './leaf_data/images/test/output'
    target_test = './leaf_data/labels/test/output'
    move_files(source_test, target_test)

    # Move validate
    source_val = './leaf_data/images/validate/output'
    target_val = './leaf_data/labels/validate/output'
    move_files(source_val, target_val)

    print("Finished moving and renaming generated files")

def move_files(source_train, target_train):
    file_names = os.listdir(source_train)
    index_gt = 44
    index_img = 36
    if "validate" in source_train:
        index_gt = 47
        index_img = 39
    if "test" in source_train:
        index_gt = 43
        index_img = 35
    for file_name in file_names:
        # Move file / and rename it
        if "groundtruth" in file_name:
            print(len(file_name))
            shutil.move(os.path.join(source_train, file_name), target_train)
            old_file = os.path.join(target_train, file_name)
            print(f"groundtruth: {file_name}")
            print(f"{file_name[(index_gt-1):]}")
            new_file_name = file_name[index_gt:]
            print(new_file_name)
            new_file = os.path.join(target_train, new_file_name)
            os.rename(old_file, new_file)
        else:
            # If this file is yet to have been name-changed
            if (len(file_name)) > 70:
                print(f"image: {file_name}")
                print(f"{file_name[(index_img-1):]}")
                old_file = os.path.join(source_train, file_name)
                new_file_name = file_name[index_img:]
                print(new_file_name)
                new_file = os.path.join(source_train, new_file_name)
                os.rename(old_file, new_file)

def createLabels(labels_src):
    index = 0
    src_files = os.listdir(labels_src)
    total = len(src_files)
    for file in src_files:
        print(f"At file: {file}, file {index} out of {total}")
        if "png" not in file:
            continue
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
    src = "leaf_data/labels/train/"
    createLabels(src)
    src = "leaf_data/labels/test/"
    createLabels(src)
    src = "leaf_data/labels/validate/"
    createLabels(src)

    # For the augmented images
    src = "leaf_data/labels/train/output/"
    createLabels(src)
    src = "leaf_data/labels/test/output/"
    createLabels(src)
    src = "leaf_data/labels/validate/output/"
    createLabels(src)

    print("Finished generating labels")

# setup creating of txts with all file name from aug files
def prepareTxtsAugment():
    # Create the train.txt and test.txt
    directory = "/leaf_data/images/train/output/"
    target_file_train = "yolov3/data/auged_leaf_train.txt"
    target_file_test = "yolov3/data/auged_leaf_test.txt"
    target_file_val = "yolov3/data/auged_leaf_val.txt"
    all_targets = [target_file_train, target_file_test, target_file_val]

    index = 0
    for targets in all_targets:
        if index == 1:
            directory = "/leaf_data/images/test/output/"
        if index == 2:
            directory = "/leaf_data/images/validate/output/"
        createTxts(directory, targets)
        index += 1

    print("Finished generating the train.txt file for the augmented images")

# Create txt files containing train / test / val file names
def prepareTxts():
    # Create the train.txt and test.txt
    directory = "/leaf_data/images/train/"
    target_file_train = "yolov3/data/leaf_train.txt"
    target_file_test = "yolov3/data/leaf_test.txt"
    target_file_val = "yolov3/data/leaf_val.txt"
    # Shapes appear to not be nec for this imp of yolo
    # shapes_target_file_train = "leaf_data/leaf_train.shapes"
    # shapes_target_file_test = "leaf_data/leaf_test.shapes"
    # shapes_target_file_val = "leaf_data/leaf_.shapes"
    # all_targets = [[target_file_train, shapes_target_file_train], [target_file_test, shapes_target_file_test],
    #                [target_file_val, shapes_target_file_val]]
    all_targets = [target_file_train, target_file_test, target_file_val]

    index = 0
    for targets in all_targets:
        if index == 1:
            directory = "/leaf_data/images/test/"
        if index == 2:
            directory = "/leaf_data/images/validate/"
        createTxts(directory, targets)
        index += 1

    print("Finished generating the train.txt files for the original images")

def createTxts(images_directory, txt_target):
    images_list_to_write = []
    # shapes_to_write = []

    for file in os.listdir('.' + images_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            images_list_to_write.append('..'+images_directory + filename)
            # image = cv2.imread(images_directory + filename)
            # Log width, height because im pretty sure that's what yolo expect
            # shapes_to_write.append([image.shape[1], image.shape[0]])

    # Save txt
    np.savetxt(txt_target, images_list_to_write, delimiter="", newline="\n",
               fmt="%s")
    # Save .shapes
    # np.savetxt(shape_target, shapes_to_write, delimiter=" ", newline="\n",
               # fmt="%s %s")

# Create yaml

if __name__ == "__main__":
    # remove_existing_files()
    # moveImagesFromOriginalDataset()
    # How many we wanna craete?
    number_auged_images_to_train = 5000
    number_auged_images_to_test = 500
    number_auged_images_to_validate = 100
    # Create and place and create txts
    createAugmentedDataset(number_auged_images_to_train, number_auged_images_to_test, number_auged_images_to_validate)
    move_Agumented()
    print("Check in console that the resulting file names are what is expected - only the last ID string of the generated file")
    setupCreateLabels()
    prepareTxts()
    prepareTxtsAugment()
