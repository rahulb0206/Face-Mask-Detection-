# Face-Mask-Detection-

This project was developed during my college days amid the COVID-19 pandemic. The aim was to create a system capable of detecting whether individuals in an image or video stream were wearing face masks or not. Such systems have become increasingly important for ensuring public safety and compliance with health regulations.

# Project Overview
The project comprises two main components: training a machine learning model to detect face masks and implementing real-time detection in a video stream.

# Training the Model
The model was trained using a dataset containing two categories of images: faces with masks and faces without masks. These images were collected and labelled to facilitate supervised learning.

The dataset used for training the model can be found in two folders:

With Mask: Contains images of faces with masks.
Without Mask: Contains images of faces without masks.
 
# Real-time Detection
The trained model is then integrated into a Python script capable of capturing video streams from a webcam. It detects faces in real-time and classifies them as either wearing a mask or not. Detected faces are marked with bounding boxes, and labels indicating the presence or absence of a mask are displayed.

# Dataset
The dataset used for training the model is crucial for the performance of the face mask detection system. It should contain a diverse range of images depicting individuals wearing masks and without masks.
![without mask screenshot](https://github.com/rahulb0206/Face-Mask-Detection-/assets/49830158/be88c10d-d0f6-4d4c-8208-4a054c87207b)

![Mask screenshot](https://github.com/rahulb0206/Face-Mask-Detection-/assets/49830158/1731c53c-3b5c-4910-b5aa-2653ff033165)

# Output
The output of the face mask detection system includes annotated images and video streams showing detected faces and their mask status.

![live without mask](https://github.com/rahulb0206/Face-Mask-Detection-/assets/49830158/de0fc6da-0ffb-4b16-bc3e-ad4b3a6bc845)
![live with mask](https://github.com/rahulb0206/Face-Mask-Detection-/assets/49830158/bd115df0-8be6-4bda-8694-b808a45b9da9)
