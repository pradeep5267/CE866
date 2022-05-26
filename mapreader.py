#!/usr/bin/env python3
# This is template for your program, for you to expand with all the correct
# functionality.
import sys
import cv2
import numpy as np

# ---------------------------------------------------------------------
# Main program.

# Ensure we were invoked with a single argument.

if len(sys.argv) != 2:
    print("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
    exit(1)

print("The filename to work on is %s." % sys.argv[1])
# ---------------------------------------------------------------------

# ------------------------------------------------------------------- #
# Name: Pradeep Kumar
# Registration ID: 2105711
# ------------------------------------------------------------------- #

def get_image(filepath):
    '''
    helper function to read image from given path
    '''
    im = cv2.imread(filepath)
    return im


def get_hsv_image(im):
    '''
    helper function to convert image from BGR to HSV color space
    '''
    img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return img_hsv


def get_north_pointer(img_hsv):
    '''
    function to extract green arrow 
    '''
    green_mask = cv2.inRange(img_hsv, (36, 25, 25), (70, 255, 255))
    north_arrow = cv2.bitwise_and(img_hsv, img_hsv, mask=green_mask)
    return north_arrow


def get_pointer_arrow(img_hsv):
    '''
    function to extract red arrow
    '''
    # since the red spectrum is found on two ends in hsv space 2 masks needs
    # to be created
    red_lower1 = np.array([0, 100, 20])
    red_upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    red_lower2 = np.array([160, 100, 20])
    red_upper2 = np.array([179, 255, 255])
    lower_red = cv2.inRange(img_hsv, red_lower1, red_upper1)
    upper_red = cv2.inRange(img_hsv, red_lower2, red_upper2)
    full_range_mask = lower_red + upper_red
    pointer_arrow = cv2.bitwise_and(img_hsv, img_hsv, mask=full_range_mask)
    return pointer_arrow


def arrange_box_points(box_points):
    '''
    converts rect box points which is ordered from bottom left in clockwise
    direction to a format required for perspective transform which is  
    3rd, 1st, 2nd, 4th starting from bottom left and moving in clockwise 
    direction code adapted from 
    https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example
    '''
    local_box = np.zeros((4, 2), dtype="float32")
    # bottom right will have the biggest value after summing
    # whereas the top left will have the smallest value
    s = box_points.sum(axis=1)
    local_box[0] = box_points[np.argmin(s)]
    local_box[2] = box_points[np.argmax(s)]
    # top right will have the biggest value after difference
    # whereas the bottom left will have the smallest value
    diff = np.diff(box_points, axis=1)
    local_box[1] = box_points[np.argmin(diff)]
    local_box[3] = box_points[np.argmax(diff)]
    # return the ordered coordinates
    return local_box


def get_perspective_crop(image, pts):
    '''
    function to crop the image using perspective transform and wrap.
    the points are of minAreaRect which is then used in perspective 
    transform and wrap to get the ROI 
    '''
    # de-structure the points
    box_pts = arrange_box_points(pts)
    (p1, p2, p3, p4) = box_pts
    # calculate width of cropped image using top right, left points
    wt_1 = np.sqrt(((p3[0] - p4[0]) ** 2) + ((p3[1] - p4[1]) ** 2))
    wt_2 = np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))
    max_wt = max(int(wt_1), int(wt_2))
    # calculate height of cropped image using bottom right, left points
    ht_1 = np.sqrt(((p2[0] - p3[0]) ** 2) + ((p2[1] - p3[1]) ** 2))
    ht_2 = np.sqrt(((p1[0] - p4[0]) ** 2) + ((p1[1] - p4[1]) ** 2))
    max_ht = max(int(ht_1), int(ht_2))
    # order points in the top-left, top-right, bottom-right, and bottom-left
    # order
    dest = np.array([
        [0, 0], [max_wt - 1, 0], 
        [max_wt - 1, max_ht - 1], [0, max_ht - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(box_pts, dest)
    cropped_image = cv2.warpPerspective(image, M, (max_wt, max_ht))
    return cropped_image


def crop_map(local_img_hsv):
    # create hsv color masks for blue color border
    blue_lower = np.array([100, 150, 0], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    mask = cv2.inRange(local_img_hsv, blue_lower, blue_upper)
    mask = np.bitwise_not(mask)
    # make a copy of the image
    mask_outmap_im = cv2.copyTo(local_img_hsv, mask)
    kernel = np.ones((5, 5), np.uint8)
    # perform erosion to remove tiny protrusions on the border
    img_erosion = cv2.erode(mask_outmap_im, kernel, iterations=1)
    img_erosion = convert_hsv_to_gray(img_erosion)
    thresh1 = threshold_im(img_erosion)
    cnt, heir = get_contours(thresh1)
    # get the biggest contour by area
    c = max(cnt, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cropped_image = get_perspective_crop(local_img_hsv, box)
    return cropped_image


def convert_hsv_to_gray(img_hsv):
    '''
    function to convert image from HSV space to 
    grayscale RGB space
    '''
    bgr_im = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    gray_im = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2GRAY)
    return gray_im


def threshold_im(im):
    '''
    function to apply thresholding operations
    '''
    blur = cv2.GaussianBlur(im, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3


def get_contours(th3):
    '''
    function to get contours
    '''
    contours, hierarchy = cv2.findContours(
        th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return(contours, hierarchy)


def get_moments(im):
    '''
    function to calculate centroid of contours
    code taken from 
    https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    '''
    M = cv2.moments(im)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


def get_min_enclosing_triangle(im):
    '''
    function to get vertices of triangle returned by
    minimum enclosing triangle
    '''
    min_enc = cv2.minEnclosingTriangle(im)
    p1, p2, p3 = min_enc[1][0], min_enc[1][1], min_enc[1][2]
    return (p1, p2, p3)


def get_triangle_base(p1, p2, p3, cX, cY):
    '''
    function to calculate the the base of triangle using euclidean 
    distance
    '''
    mid_dict = {}
    base_dict = {}

    # get the distance of each side then get the minimum distance
    # which will be the base of the triangle
    # then calculate the mid point of the base
    d1 = cv2.norm(np.array([p1[0][0], p1[0][1]]) -
                  np.array([p2[0][0], p2[0][1]]), cv2.NORM_L2)
    d2 = cv2.norm(np.array([p2[0][0], p2[0][1]]) -
                  np.array([p3[0][0], p3[0][1]]), cv2.NORM_L2)
    d3 = cv2.norm(np.array([p3[0][0], p3[0][1]]) -
                  np.array([p1[0][0], p1[0][1]]), cv2.NORM_L2)
    base_dict[d1] = [np.array([p1[0][0], p1[0][1]]),
                     np.array([p2[0][0], p2[0][1]])]
    base_dict[d2] = [np.array([p2[0][0], p2[0][1]]),
                     np.array([p3[0][0], p3[0][1]])]
    base_dict[d3] = [np.array([p3[0][0], p3[0][1]]),
                     np.array([p1[0][0], p1[0][1]])]
    a, b = base_dict[min(base_dict.keys())]
    a2, b2 = ((a[0] + b[0])/2, (a[1] + b[1])/2)

    m1 = cv2.norm(np.array([p1[0][0], p1[0][1]]) -
                  np.array([cX, cY]), cv2.NORM_L2)
    m2 = cv2.norm(np.array([p2[0][0], p2[0][1]]) -
                  np.array([cX, cY]), cv2.NORM_L2)
    m3 = cv2.norm(np.array([p3[0][0], p3[0][1]]) -
                  np.array([cX, cY]), cv2.NORM_L2)

    # calculate the distance between the 3 vertices of the triangle and 
    # centroid of the triangle
    # the largest distance will be between the tip of the triangle and the
    # centroid of the triangle since its an isoceles triangle
    mid_dict[m1] = [np.array([p1[0][0], p1[0][1]]), np.array([cX, cY])]
    mid_dict[m2] = [np.array([p2[0][0], p2[0][1]]), np.array([cX, cY])]
    mid_dict[m3] = [np.array([p3[0][0], p3[0][1]]), np.array([cX, cY])]
    d_mid, c_mid = mid_dict[max(mid_dict.keys())]
    return (a2, b2, d_mid, base_dict)


def check_orientation(th3):
    '''
    helper function to get the points for base of the triangle 
    and centroid which will then be used to check the orientation 
    of the green (north) pointer arrow
    '''
    north_arrow_contours, hierarchy = get_contours(th3)
    cX, cY = get_moments(north_arrow_contours[0])
    p1, p2, p3 = get_min_enclosing_triangle(north_arrow_contours[0])
    a2, b2, c, base_dict = get_triangle_base(p1, p2, p3, cX, cY)
    return (a2, b2, cX, cY, c, base_dict)


def get_angles(x1, y1, x2, y2):
    '''
    function to calculate the tan inverse of the slope of the 
    line drawn from centroid to tip of the red triangle
    '''
    m = ((int(x2) - int(y2) / (int(x1) - int(y1))))
    a_t = np.arctan2((int(x2) - int(y2)), (int(x1) - int(y1)))
    return (m, a_t)

# driver code
im = get_image(sys.argv[1])

img_hsv = get_hsv_image(im)
north_arrow_cropped_map = img_hsv.copy()
red_arrow_cropped_map = img_hsv.copy()

north_arrow = get_north_pointer(north_arrow_cropped_map)
north_arrow_bgr = convert_hsv_to_gray(north_arrow)

th3 = threshold_im(north_arrow_bgr)
a2, b2, cX, cY, c, base_dict = check_orientation(th3)

# as the min enclosing triangle returns an inverted triangle for the 
# green pointer arrow, the required orientation is achieved by rotating 
# the image by 90 degrees clockwise till the value of y co ordinate of 
# the base of the triangle is greater than the centroid's y co ordinate
while(True):
    if (int(b2**2) > int(cY**2)):
        th3 = cv2.rotate(th3, cv2.ROTATE_90_CLOCKWISE)
        img_hsv = cv2.rotate(img_hsv, cv2.ROTATE_90_CLOCKWISE)
        a2, b2, cX, cY, c, base_dict = check_orientation(th3)
    else:
        break
img_hsv = crop_map(img_hsv)
cropped_map = img_hsv.copy()

pointer_arrow = get_pointer_arrow(cropped_map)
pointer_arrow_bgr = convert_hsv_to_gray(pointer_arrow)
th4 = threshold_im(pointer_arrow_bgr)
rp_a2, rp_b2, rp_cX, rp_cY, rp_c, base_dict = check_orientation(th4)

x, y, c = img_hsv.shape
m, a_t = get_angles(rp_c[0], rp_cX, rp_c[1], rp_cY)

# x co ordinate needs to be adjusted since the image is stored in matrix form 
# where the index starts from top left corner however the map's co ordinates 
# start from bottom left corner
print(f'POSITION {rp_c[0]/y} {(x - rp_c[1])/x}')
# +90 degree correction since the bearing has to calculated wrt to true north
print(f'BEARING {np.degrees(a_t)+90}')
