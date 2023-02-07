import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import defaultdict
import math

roi_defined = False
threshold = 20

def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked,
	# record the starting ROI coordinates
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)
		roi_defined = True

def calculate_gradient_orientation(frame, threshold):
    """
    This function calculates the gradient orientation and gradient magnitude 
    for a given grayscale image.

    Parameters:
    - frame (ndarray): Input grayscale image
    - threshold (float): Minimum gradient magnitude to consider a pixel for further processing

    Returns:
    - tuple: Tuple containing gradient magnitude, gradient orientation, valid orientation pixels, 
             invalid orientation pixels and valid index.
    """
    # Calculate gradient magnitude using gradient of image
    gradient_x, gradient_y = np.gradient(frame[:, :, 2])
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Calculate gradient orientation using arctan2
    gradient_orientation = np.arctan2(gradient_x, gradient_y)


    # Identify invalid pixels with gradient magnitude below the threshold
    invalid_index = np.where(gradient_magnitude < threshold)
    valid_index = np.where(gradient_magnitude > threshold)

    # Normalize orientation values and convert to uint8
    valid_orientation = cv2.cvtColor(np.float32(gradient_orientation), cv2.COLOR_GRAY2BGR)
    valid_orientation = cv2.normalize(valid_orientation, None, 0, 255, cv2.NORM_MINMAX)
    valid_orientation = np.uint8(valid_orientation)
    valid_orientation[invalid_index[0], invalid_index[1], :] = [0, 0, 255]

    gradient_magnitude = (gradient_magnitude-gradient_magnitude.min())/(float)(gradient_magnitude.max() - gradient_magnitude.min())

    return gradient_magnitude, gradient_orientation, valid_orientation, invalid_index, valid_index

def calcHoughTransform(accumulator, angle, radii_table, valid_indices):
    """
    Calculate the Hough transform given the orientation and radius information.
    
    Parameters:
    - accumulator: 2D numpy array representing the accumulator space
    - angle: 2D numpy array representing the gradient orientation
    - radii_table: dictionary representing the possible radii values
    - valid_indices: tuple of two 1D numpy arrays representing the x and y indices of valid pixels

    Returns:
    - accumulator: updated accumulator space
    """
    
    # Convert orientation to degrees and round to nearest integer
    angle_degrees = np.round(np.rad2deg(angle)).astype(np.int32)
    
    # Create a kernel for each radius value
    for radius, values in radii_table.items():
        for value in values:
            x = valid_indices[1] + value[0]
            y = valid_indices[0] + value[1]
            
            # Check if the indices are within bounds
            mask = np.logical_and(
                np.logical_and(x >= 0, x < accumulator.shape[1]),
                np.logical_and(y >= 0, y < accumulator.shape[0]),
            )
            
            # Increment the accumulator for the corresponding angle and radius
            accumulator[y[mask], x[mask]] += 1
    
    return accumulator

cap = cv2.VideoCapture('video\VOT-Woman.mp4')

# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break

track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+w, r:r+h]
# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# computation mask of the histogram:
# Pixels with S<30, V<20 or V>235 are ignored
mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
# Marginal histogram of the Hue component
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )


# Create the R-table to store the distances of valid gradient pixels based on their orientation
r_table = defaultdict(list)

# Calculate the gradient magnitude and orientation
gradient, orientation, _, _, valid_indices = calculate_gradient_orientation(hsv_roi, threshold)

# Find the center of the ROI
roi_center = np.array([int(r + (h//2)), int(c + (w//2))])

# Convert orientation values from radians to degrees
orientation = np.round(orientation * 180 / np.pi).astype(np.int32)

# Populate the R-table
for px, py in zip(valid_indices[0], valid_indices[1]):
    # Calculate the distance from the center of the ROI for each valid gradient pixel
    distance = roi_center - np.array([py + r, px + c])
    # Store the distance in the R-table based on the orientation of the pixel
    r_table[orientation[px, py]].append(distance)

cpt = 1
while(1):
	ret ,frame = cap.read()
	if ret == True:

		frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		# Calculate gradient magnitude and orientation
		gradient, orientation, ori, _, valid_indices = calculate_gradient_orientation(frame_hsv, threshold)

		# Initialize Hough Transform
		hough_transform = np.zeros(orientation.shape)

		# Calculate Hough Transform
		hough_transform = calcHoughTransform(hough_transform, orientation, r_table, valid_indices)


		#mean shift
		ret, track_window = cv2.meanShift(hough_transform, track_window, term_crit)
		r,c,h,w = track_window

		# Draw a blue rectangle on the current image and normalize Hough
		frame_tracked = cv2.rectangle(frame, (r, c), (r + h, c + w), (255, 0, 0), 2)
		# Normalize Hough Transform
		hough_transform = (hough_transform-hough_transform.min())/(float)(hough_transform.max() - hough_transform.min())

		#Plotting all images
		cv2.imshow('Sequence', frame_tracked)
		cv2.imshow('Orientation', ori)
		cv2.imshow("Transformee Hough", hough_transform)

		k = cv2.waitKey(60) & 0xff
		if k == 27:
				break
		elif k == ord('s'):
				cv2.imwrite('./images/Q5_Frame_%04d.png'%cpt,frame_tracked)
				cv2.imwrite('./images/Q5_Frame_tHough_%04d.png'%cpt,hough_transform)
				cv2.imwrite('./images/Q5_Frame_Ori_%04d.png'%cpt,ori)
		cpt += 1
	else:
		break

cv2.destroyAllWindows()
cap.release()