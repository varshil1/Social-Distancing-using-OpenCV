# Social-Distancing-using-OpenCV

### Introduction
Controlling the spread of contagious diseases is done through social distance.<br/>
As the term implies, social distancing entails people physically separating themselves from one another in order to reduce intimate contact and hence the spread of an infectious disease (such as coronavirus).<br/>
But as seen at public places, breaching of social distancing regulations are observed more often.So what if we can built a software which can detect this violation and send warning to the authority to take necessary steps.

### Aim
The main aim of this project is to monitor social distancing through human detection for preventing/reducing COVID spread using modern technologies.

## Solution
The video from the cctv camera can be processed using deep learning in order to detect the people . The distance between people can be calculated in order to find whether the social distancing is maintained or not.

## Steps
* Load video frame
* Detect number of person using YOLO
* Measure the distance between each person
* If Distance measure < Minimum distance : Red box
* If Distance measure > Minimum distance , but < Maximum distance : Yellow box
* If Distance measure >> Minimum distance : Green box
* Calculate number of violations at particular time step.
* If violation > Threshold/allowed violation: Send Email.

The video is converted to a bird's eye view and sent into the YOLOv3 model, which has already been trained to detect objects.
The Common Object in Context is used to train the YOLOv3 model (COCO). A pre-recorded video supported the suggested system. 
The system's results and outcomes reveal that detecting whether or not regulations are broken requires evaluating the distance between different individuals. 
Individuals are represented by a red bounding box if the distance is less than the minimal threshold value; otherwise, they are represented by a green bounding box.


### Results
> Input

![This is an input image](videos/input.gif)
> Output

![This is an output image](videos/output.gif)
### References 

* https://iopscience.iop.org/article/10.1088/1742-6596/1916/1/012039/pdf

* https://ieeexplore.ieee.org/document/9243478

* https://link.springer.com/article/10.1007/s41870-021-00658-2
