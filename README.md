# Homography-Estimation

The objective is to perform camera pose estimation using homography. Given a video the
task is to compute the rotation and translation between the camera and a coordinate frame whose
origin is located on any one corner of the sheet of paper.

Steps involved:
1. Designing an image processing pipeline to extract the paper on the ground and then extract all
of its corners using the Hough Transformation technique.
2. Once you have all the corner points, computing homography between real
world points and pixel coordinates of the corners. 
3. Decomposing the obtained homography matrix to get the rotation and translation

Data:
The dimensions of the paper is 21.6 cm x 27.9 cm.
