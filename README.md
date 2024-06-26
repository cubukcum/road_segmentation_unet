# Road Segmentation with UNet

![Model Test Results](./resnet50.JPG)

This repository contains code for training and evaluating a UNet-based segmentation model for road image segmentation task. The model is implemented using the segmentation-models library and is designed to segment images into five classes: Road, Lane, Undrivable, Movable, and Car.

Segmentation Models library was used with Resnet50 backbone and U-Net to achieve better performance. Model was trained for 40 epochs and then converted to tflite model to use on Raspberry Pi.

## Training Process

The model was trained using a dataset consisting of annotated road images. Training was conducted for 40 epochs to ensure convergence and achieve optimal performance. During training, the model learns to accurately classify pixels into the predefined classes, enabling it to effectively segment road scenes under various conditions, including different lighting, weather, and road surface types.

## Usage

Install necessary packages with sh setup.sh
You may need to change the code depending on your input choice of camera. Also feel free to adjust display_mode, num_threads, enable_edgetpu, camera_id, width, height parameters to find what fits well for you.

## Model Deployment

Upon training completion, the model was converted to a TensorFlow Lite format for efficient deployment on resource-constrained platforms such as the Raspberry Pi. This lightweight model enables real-time segmentation inference, making it suitable for embedded applications in autonomous vehicles or edge computing scenarios.
