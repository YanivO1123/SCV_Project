# SCV_Project

In order to use / evaluate our work, a clear description of our contribution / code is provided below.

## Demo
One can use the images in the directory `demo` for a short demo. The images includes a variety of images, including plant images the network has not trained on.

```
python ./yolov3/detect.py --weights ./yolov3/runs/train/exp/trained_weights.pt --source ./leaf_data/demo_images --img-size 416
```

The defaults have been pre-configured along with a path to two example image files.

## Setup data
In the `create_dataset.py` file, we call the following method to prepare the data for training.
0. Create a directory called `leaf_data`, which includes two directories `images` and `labels`, which again both include three directories `train`, `validate`, and `test`.
1. Call `moveImagesFromOriginalDataset()` to copy the dataset images into the network training directory.
2. Call `createAugmentedDataset(num_train, num_test, num_val)` in order to create the augmented images. *Created by Yaniv*
3. Call `moveAugmented()` to then move the augmented label-images (the images based on which we create the bounding boxes) to the label images directory. *Created by Yaniv & Sterre*
4. Call `setupCreateLabels()` to create the bounding boxes txt files, for all un-augmented images, in the right directory (along the label-images). *Created by Yaniv & Sterre*
5. Call `setupCreateLabelsAugmented()` to create the bounding boxes txt files, for all augmented images, in the right directory (along the label-images). *Created by Yaniv & Sterre*
6. Call `prepareTxts()` to create a train.txt, test.txt and val.txt, which contain a list of the files to be used in the network training for train / test / validation. *Created by Yaniv & Sterre*  
7. Call `prepareTxtsAugment()` to do the same for the augmented images. *Created by Yaniv & Sterre*

## Test bounding boxes:
In the file `show_bounding_boxes.py` we have implemented functionality to show bounding boxes, for bounding box testing purposes. *Created by Sterre*

## Train
In order to train, we have imported from https://github.com/ultralytics/yolov3.git, created a dataset yaml file (yolov3/data/leaf.yaml) which specifies the dataset information (number of classees, class names, source for train / test / validation).

To train the network, we call: 

```
python train.py --weights yolov3-tiny.pt --data data/leaf.yaml --img-size 416 --epochs 60
```

In order to avoid a large number of passed parameters, all of the above have been changed into the parsing-default, so one can also only call: python train.py

The results, along with the trained weights, are automatically generated into runs/train/expX.

## Detection
In order to detect, we call: 

```
python ./yolov3/detect.py --weights path_to_weights --source images_to_detect_on --img-size 416
```
The trained weights are available in the path "./yolov3/runs/train/exp/trained_weights.pt"

## Additional files
`YOLOv3_tutorial_based.py` and `util.py` are a yolo-architecture and detector implementation based on https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch
