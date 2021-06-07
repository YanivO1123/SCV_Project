# SCV_Project

In order to use / evaluate our work, a clear description of our contribution / code is provided below: 

## Setup data:
In the create_dataset.py file, we first call moveImagesFromOriginalDataset() to copy the dataset images into the netowrk training directory.
1. We call: createAugmentedDataset(num_train, num_test, num_val) in order to create the augmented images. *Created by Yaniv*
2. We call: move_Agumented() to then move the augmented label-images (the images based on which we create the bounding boxes) to the label images directory. *Created by Yaniv & Sterre*
3. We call: setupCreateLabels() to create the bounding boxes txt files, for all un-augmented images, in the right directory (along the label-images). *Created by Yaniv & Sterre*
4. We call: setupCreateLabelsAugmented() to create the bounding boxes txt files, for all augmented images, in the right directory (along the label-images). *Created by Yaniv & Sterre*
5. We call: prepareTxts() to create a train.txt, test.txt and val.txt, which contain a list of the files to be used in the network training for train / test / validation. *Created by Yaniv & Sterre*  
6. We call: prepareTxtsAugment() to do the same for the augmented images. *Created by Yaniv & Sterre*

## Test bounding boxes:
In the file plotBoundingBoxes.py we have implemented functionality to show bounding boxes, for bounding box testing purposes. *Created by Sterre*

## Train:
In order to train, we have imported from https://github.com/ultralytics/yolov3.git, created a dataset yaml file (yolov3/data/leaf.yaml) which specifies the dataset information (number of classees, class names, source for train / test / validation).

To train the network, we call: python train.py --weights yolov3-tiny.pt --data data/leaf.yaml --img-size 416 --epochs 60

In order to avoid a large number of passed parameters, all of the above have been changed into the parsing-default, so one can also only call: python train.py

The results, along with the trained weights, are automatically generated into runs/train/expX.

## Detection:
In order to detect, we call: python detect.py --weights path_to_weights --source images_to_detect_on --img-size 416

Or simply, python detect.py (the defaults have been pre configured as for the above).

## Additional files:
YOLOv3_tutorial_based.py and util.py are a yolo-architecture and detector implementation based on https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch
