from video_magnifier import turnVideoIntoListOfFrames, viewFrame
import random
import imageio
import pickle
from video_processor import batchAndDifferentiate
import numpy as np
import cv2
from image_distortion_simulator import imageify
from collections import defaultdict


path = "IMG_0478.m4v"

PICKLE_DUMP = False
DUMP_HANDFUL_OF_FRAMES = False
DOWNSAMPLE_IMAGES = False
SPLAY_IMAGE = False
DETECT_VANISHING_POINTS = True

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections

def detectVanishingPoints(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.medianBlur(gray, 5)
	adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
	thresh_type = cv2.THRESH_BINARY_INV
	bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2)
	rho, theta, thresh = 2, np.pi/180, 400
	lines = cv2.HoughLines(bin_img, rho, theta, thresh)
	segmented = segment_by_angle_kmeans(lines)
	intersections = segmented_intersections(segmented)

	return intersections

if PICKLE_DUMP:

	vid = imageio.get_reader(path,  'ffmpeg')



	listOfFrames = turnVideoIntoListOfFrames(vid, 1000, 1500)

	print "pickling..."

	pickle.dump(listOfFrames, open("jumpy_video_1000_1500.p", "w"))

	for _ in range(100):
		viewFrame(random.choice(listOfFrames))

if DUMP_HANDFUL_OF_FRAMES:
	listOfFrames = pickle.load(open("jumpy_video_0_500.p", "r"))

	pickle.dump(listOfFrames[:20], open("jumpy_video_0_20.p", "w"))

if DOWNSAMPLE_IMAGES:
	listOfFrames = pickle.load(open("jumpy_video_0_20.p", "r"))

	batchedVid = batchAndDifferentiate(np.array(listOfFrames), \
		[(1, False), (5, False), (5, False), (1, False)])

	pickle.dump(batchedVid, open("jumpy_video_0_20_batched.p", "w"))

if SPLAY_IMAGE:
#	listOfFrames = pickle.load(open("jumpy_video_0_20_batched.p", "r"))

	vid = imageio.get_reader(path,  'ffmpeg')

	frame = vid.get_data(150)

#		diffFrame = batchAndDifferentiate(frame, [(1, True), (1, True), (1, False)])

#		viewFrame(np.abs(diffFrame), magnification=1e1)

	img = frame

#		gray = np.average(img, axis=2)

#		viewFrame(imageify(gray)/255, magnification=1)

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 



	cv2.imwrite('gray.jpg', gray) 
	  
	# Apply edge detection method on the image 
	edges = cv2.Canny(gray,50,150,apertureSize = 3) 

	cv2.imwrite('edges.jpg', edges) 
	  
	# This returns an array of r and theta values 
	lines = cv2.HoughLines(edges,1,np.pi/180, 100) 
	  
	# The below for loop runs till r and theta values  
	# are in the range of the 2d array 
	for line in lines:
		for r,theta in line: 
		      
		    # Stores the value of cos(theta) in a 
		    a = np.cos(theta) 
		  
		    # Stores the value of sin(theta) in b 
		    b = np.sin(theta) 
		      
		    # x0 stores the value rcos(theta) 
		    x0 = a*r 
		      
		    # y0 stores the value rsin(theta) 
		    y0 = b*r 
		      
		    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
		    x1 = int(x0 + 1000*(-b)) 
		      
		    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
		    y1 = int(y0 + 1000*(a)) 
		  
		    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
		    x2 = int(x0 - 1000*(-b)) 
		      
		    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
		    y2 = int(y0 - 1000*(a)) 
		      
		    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
		    # (0,0,255) denotes the colour of the line to be  
		    #drawn. In this case, it is red.  
		    cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2) 
	      
	# All the changes made in the input image are finally 
	# written on a new image houghlines.jpg 
	cv2.imwrite('linesDetected.jpg', img) 

if DETECT_VANISHING_POINTS:

	vid = imageio.get_reader(path,  'ffmpeg')

	img = vid.get_data(0)

	vanishingPoints = detectVanishingPoints(img)

	print len(vanishingPoints)

	for vanishingPoint in vanishingPoints:
		cv2.point(img, vanishingPoint)

	cv2.imwrite('pointsDetected.jpg', img)