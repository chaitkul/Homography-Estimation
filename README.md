# Homography-Estimation

The objective is to perform camera pose estimation using homography without using inbuilt OpenCV functions. Given a video the
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


![Homography Screenshot](https://github.com/chaitkul/Homography-Estimation/assets/127642282/35075e6e-171d-46e4-b964-b3e1510e5368)

The Rotation matrix, Translation vector and the Homography matrix will be printed at each iteration

![Homography Screenshot Terminal](https://github.com/chaitkul/Homography-Estimation/assets/127642282/1ebfee05-6c3e-4092-967b-acd2bcfbc944)
