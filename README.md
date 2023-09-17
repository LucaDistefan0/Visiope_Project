# Visiope_Project
Computer Vision on Embedded Devices - Object Recognition and Energy analysis

Object detection is widely used for face detection, vehicle detection, pedestrian counting, web images, security systems and self-driving cars. In this project, we’ll work with the Freenove Robot Dog Kit for Raspberry Pi 4, make it explore the environment and perform object recognition using the camera already on board.

We’ll use object detection-algorithms and methods based on deep learning. To achieve our goal we’ll use a lightweight neural network (LWN).

We want to detect each and every object in image by the area object in an highlighted rectangular boxes, identify each and every object and assign its tag them. Also, we are interested in an energy drain analysis in order to optimize its usability while powered only by batteries.

The dataset that will are considering is COCO (Common Objects in Context). The COCO train, validation, and test sets, containing more than 200,000 images and 80 object categories.

Source: https://cocodataset.org/

UPDATE:
- "Object_detection.ipynb" for the model training and test for classification purpose. It'll be converted then in a tensorflow lite model.
- "object_detection_tflite.py" for the execution of the detection on the Raspberry Pi 4 using the tflite converted model.
