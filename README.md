# Computer-Vision-Projects-
List of computer vision projects


## Application to detect face, smile, blink and wink of a person

##### Project approach:

Build a standalone windows application file which gets feed from the webcam and shows whether the person in the frame smiles or not along with blink and wink counters. Prepared a simple GUI that shows video feed in one frame and necessary counters in another frame. It has reset and quit buttons to reset the counters to zero and quit the application respectively.

Pre-trained model in DLIB library is used to get the 68 facial landmark points. Among which only eye and mouth points are taken into consideration for this project. Frames from the video is processed to show the face bounding box over faces identified, eyes and lips contours over the frame image.

Smile detection is done by calculating the SAR (smile aspect ratio) which is the ratio of length of the mouth to length of jaw. If a person smiles, mouth length increases and hence ratio increases. This helps to capture whether a person is smiling or not if the ratio is more than the threshold value.

Eye blink is detected by calculating the EAR (Eye aspect ratio) which is the ratio of two vertical lengths between lower and upper eye lids to length of each eye. EAR is average value of two eyes. If a person blinks, vertical measurements decreases and hence the ratio decreases to a great extent. This helps to capture whether a person is blinking or not if the ratio is less than the threshold value. Additionally, blink is considered as happened only when the three consecutive frames got less than threshold value.

Similarly, right and left wink is detected using the EAR. In this case, respective eye wink is considered happened only when the specific eye EAR value is less than the threshold value in three consecutive frames and EAR of another eye.

Programming language: Python

Libraries used: cv2, Dlib, imutils, PIL, scipy, time, tkinter, pyinstaller



## Build a machine learning model to predict the gender based on user given names.

##### Dataset: 
Given CSV file contains names and its respective gender.

##### Project Approach: 
Given name is tokenized into characters and fed into a neural network that gives the output of gender probability. Gender is predicted based on the probability threshold value of 0.5.

Programming language: Python

Libraries/Frameworks used: Tensorflow, numy, pandas, matplotlib


## Real time video stream object detection using YOLO

##### Project Approach: 

Build a real time object detection using YOLO model on COCO dataset. This dataset can detect around 80 common objects. 


Programming language: Python

Libraries used: cv2, numpy

