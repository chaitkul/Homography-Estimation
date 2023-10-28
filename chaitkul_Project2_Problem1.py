# QUESTION 1 ----------------------------------------------------------------------------------------

import cv2
import numpy as np
import os
from scipy.spatial.transform._rotation import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Creating a video path and opening the video file

CURRENT_DIR = os.path.dirname(__file__)
video_path = os.path.join(CURRENT_DIR,'project2.avi')
paper_video = cv2.VideoCapture(video_path)

roll1 = []
pitch1 = []
yaw1 = []
x1 = []
y1 = []
z1 = []

# Print the error message if there is any issue in opening the video

if (paper_video.isOpened() == False):
    print("Error opening the video file")

# Opening the frames in the video

while(paper_video.isOpened()):
    ret, frame = paper_video.read()
    fps = paper_video.get(5)

    if ret == True:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)          # Converting the frame from BGR to HSV
        key = cv2.waitKey(50)
        
        if key == ord('q'):
            break

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting the frame from BGR to Gray
        gaussian = cv2.GaussianBlur(frame, (11,11), 0)        # Applying gaussian blur to the image with kernel size of 11
        canny_image = cv2.Canny(gaussian,150,200)             # Using Canny Edge detection method to get the periphery of the paper

        # cv2.imshow("canny image", canny_image)

        # Getting the dimensions of the image and setting the maximum values for r and theta
        # r is the distance from the linr to origin
        # theta is the angle made by the line with positive x axis

        rows, columns = canny_image.shape                        
        r_max = int(np.sqrt(rows**2 + columns**2))            
        theta_max = 360
        votes = np.zeros((r_max,theta_max))

        # Function to calculate hough transform

        def hough_transform(canny_image):
            for y in range(rows):                     # Iterating through the video to find the non zero pixels from the canny image
                for x in range(columns):
                    if canny_image[y,x] != 0:
                        for theta in range(theta_max):      
                            r = int(x*np.cos(np.deg2rad(theta)) + y*np.sin(np.deg2rad(theta)))  # Getting the respective values for r and theta where edge is detected
                            votes[r,theta] = votes[r,theta] + 1   # Incrementing the vote value everytime a nonzero value pixel is detected

            threshold = 145                           # Setting a threshold value to get only the strong lines
            lines = []

            for vote in np.argwhere(votes>threshold): # Only the votes that are greater than the threshold value are considered.
                r = vote[0]
                theta = vote[1]
                a = np.cos(np.deg2rad(theta))        
                b = np.sin(np.deg2rad(theta))
                x0 = a * r                            # x = r*cos(theta) and y = r*sin(theta)
                y0 = b * r
                x1 = int(x0 + 2000 * (-b))            # Extending the lines with the help of pixels 
                y1 = int(y0 + 2000 * (a))
                x2 = int(x0 - 2000 * (-b))
                y2 = int(y0 - 2000 * (a))
                lines.append([x1,y1,x2,y2])           # Appending the lines found in a list
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)     # Drawing the lines onto the frame

            global corner_points
            corner_points = []                        
            for i in range(len(lines)):               # Finding the corners from the intersection points of the lines
                for j in range(len(lines)):
                    x1,y1,x2,y2 = lines[i]
                    x3,y3,x4,y4 = lines[j]
                    slope1 = (y2-y1)/(x2-x1)          # Solving the equations of lines simultaneously to get the corner
                    slope2 = (y4-y3)/(x4-x3)
                    intercept1 = y1-slope1*x1
                    intercept2 = y3-slope2*x3
                    try:
                        x = (intercept2-intercept1)/(slope1-slope2) # FInding the x coordinate
                        y = slope1*x+intercept1                     # Finding the y coordinate
                        if x>0 and x<frame.shape[1] and y>0 and y<frame.shape[0]:      # Checking if the point found is within the bounds of the frame
                            cv2.circle(frame, (int(x),int(y)), 5, (0,255,0), -1)       # Drawing a circle at the corner point
                            corner_points.append((int(x),int(y)))                      # Appending the corner points into a list
                    except:
                        pass
            
            return lines, corner_points
        
        # Applying the hough transform function to the canny image

        hough_transform(canny_image)

        # Function to find the homography matrix

        def homography(pixel_coordinates, real_coordinates):    
            K = np.array([[1382.58398, 0, 945.743164],      # The intrinsic matrix of the camera
            [0, 1383.57251, 527.04834],
            [0, 0, 1]])

            A = np.zeros((real_coordinates.shape[0]+pixel_coordinates.shape[0],9))  # Creating a matrix with a size of 8*9 for four corner points
            try:
                for i in range(real_coordinates.shape[0]):

                    # Updatng the A matrix to solve for homography

                    x, y = pixel_coordinates[i]             
                    X, Y = real_coordinates[i]
                    A[2*i] = [x,y,1,0,0,0,-X*x,-X*y,-X]
                    A[2*i+1] = [0,0,0,x,y,1,-Y*x,-Y*y,-Y]
            except:
                pass

            AtA = np.dot(A.T,A)                                                 # Multiplying A' by A
            eigen_values, eigen_vectors = np.linalg.eig(AtA)                    # Finding the eigen values and eigen vectors of A'A 
            min_value_eigen_vector = eigen_vectors[:,np.argmin(eigen_values)]   # Getting the minimum value eigen vector
            H = min_value_eigen_vector.reshape((3,3))                           # Finding H corresponding to the minimum value eigrn vector
            H = H/H[2,2]                                                        # Normalizing H so that the last element is 1

            K_inverse = np.linalg.inv(K)                                        # Multiplying K_inverse to H
            RT = np.dot(K_inverse,H)
            return RT
        
        real_coordinates = np.array([(0, 0), (21.6, 0), (0, 27.9), (21.6, 27.9)])
        pixel_coordinates = np.array((corner_points))

        RT = homography(real_coordinates,pixel_coordinates)

        # Function to decompose homography

        def decompose_homography(H):
            try:
                U, S, V = np.linalg.svd(H)      # Decomposing the homography matrix using singular value decomposition
                    
                if U[2, 2] < 0:                
                    U *= -1
                    V *= -1

                global R, T
                R = np.dot(U, V)                        # Finding the rotation matrix
                T = H[:, 2] / np.linalg.norm(H[:, 0])   # Finding the translation matrix

                return R,T

            except:
                return None

        try:
            R,T = decompose_homography(RT)         # Decomposing hommography matrix if it is not None
            print()
            print("Rotation matrix")
            print(R)                               # Printing the homography, rotation and translation matrices
            r = Rotation.from_matrix(R)
            roll,pitch,yaw = r.as_euler('xyz', degrees = True)
            roll1.append(roll)                     # Appending the roll, pitch and yaw into the list
            pitch1.append(pitch)
            yaw1.append(yaw)
            print()
            print("Translation matrix")            # Appending x, y, and z into a list
            print(T)
            x1.append(T[0])
            y1.append(T[1])
            z1.append(T[2])

            print()
            print("Homography matrix")
            print()
            print(RT)
        except:
            pass

        cv2.imshow("frame", frame)

    else:
        break

# Release the video
paper_video.release()
cv2.waitKey(100)
cv2.destroyAllWindows()

# Plotting x, y and z 

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x1, y1, z1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')

# Plotting roll, pitch and yaw

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(roll1, pitch1, yaw1)
ax.set_xlabel('Roll')
ax.set_ylabel('Pitch')
ax.set_zlabel('Yaw')
ax.set_title('3D Scatter Plot')

