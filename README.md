# Face-Mask-Detection-

This project was developed during my college days amid the COVID-19 pandemic. The aim was to create a system capable of detecting whether individuals in an image or video stream were wearing face masks or not. Such systems have become increasingly important for ensuring public safety and compliance with health regulations.

# Project Overview
The project comprises two main components: training a machine learning model to detect face masks and implementing real-time detection in a video stream.

# Training the Model
The model was trained using a dataset containing two categories of images: faces with masks and faces without masks. These images were collected and labeled to facilitate supervised learning.

The dataset used for training the model can be found in two folders:

With Mask: Contains images of faces with masks.
Without Mask: Contains images of faces without masks.
 
# Real-time Detection
The trained model is then integrated into a Python script capable of capturing video streams from a webcam. It detects faces in real-time and classifies them as either wearing a mask or not. Detected faces are marked with bounding boxes, and labels indicating the presence or absence of a mask are displayed.

# Dataset
The dataset used for training the model is crucial for the performance of the face mask detection system. It should contain a diverse range of images depicting individuals wearing masks and without masks. The dataset used in this project can be found in the respective folders mentioned above.

# Output
The output of the face mask detection system includes annotated images and video streams showing detected faces and their mask status.
