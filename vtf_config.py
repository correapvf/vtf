import numpy as np
import cv2

#### define variables
display = True # display frames while executing
save_format = 'gray' # how to save frames, options are:
# gray_norm - gray-scaled frame with normalized brightness and contrast
# gray - gray-scaled frame (use only green channel)
# color - colored frame
# color_all - complete colored frame, except the top overlay

overlay_top = 120 # number of top pixels to cut the overlay
out_W, out_H = 1024, 768 # output keyframe width and height
overlap = 0.05 # percent one keyframe should overlap another
time_offset = 1 # set offset in seconds

blur_thr = 1500 # below this value frame is considered too bluried
blur_skip = 8 # skip n consecutive frames when blur is detected

# maximum time allowed for new keyframe.
# if this time is reached, a frame is always grabbed
elapsed_max = 60 # in seconds


#### Define a function to detect overlay
"""
check_overlay must be a function that input the frame and output a True/False value,
indicating that must be skipped (True) or processed (False)
You can change this example below that I used in my videos
You have to select a feature in the overlay that can easily be compared, and check if it match
values that you previously saved or that you know (eg. all values in one area should be 0)
If you don't have overlays in your videos, you can simply put
def check_overlay(img):
    return False
"""

sony1_overlay = np.load('overlay/sony1_overlay.npy')
sony2_overlay = np.load('overlay/sony2_overlay.npy')
sony3_overlay = np.load('overlay/sony3_overlay.npy')
feed_overlay = np.load('overlay/feed_overlay.npy')

def check_overlay(img):
    overlay_check1 = img[126:134, 40:140, 1]
    overlay_sobel = cv2.Sobel(overlay_check1, cv2.CV_16S, 0, 1, ksize=3)
    sony1_dif = np.sum(np.abs(overlay_sobel - sony1_overlay))
    sony3_dif = np.sum(np.abs(overlay_sobel - sony3_overlay))

    overlay_check2 = img[578:588, 460:500, 1]
    sony2_dif = np.sum(np.abs(overlay_check2 - sony2_overlay))

    feed_dif = np.sum(np.abs(overlay_check2 - feed_overlay))

    return sony1_dif < 30000 or sony3_dif < 30000 or sony2_dif < 10 or feed_dif < 10


#### advanced settings that you can change as well
# values to normalize image by 127 mean and variance below
norm_var = 1200 # variance to normalize image 
min_contrast = 0.7 # maximun allowed contrast
max_contrast = 2.0 # minimun allowed contrast

points_to_LK = 300 # minimum points to use LK. below that will use Farneback
key_move_thr = 150 # number of pixels the edge of one key frame can move compared to the last
downsize_interpolate = 8 # downscale frame by value to calculate brightest spot  

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners = 800, # How many pts. to locate
                      qualityLevel = 0.1, # b/w 0 & 1, min. quality below which everyone is rejected
                      minDistance = 20, # Min eucledian distance b/w corners detected
                      blockSize = 5) # Size of an average block for computing a derivative covariation matrix over each pixel neighborhood

# Parameters for lucas kanade optical flow
lk_params = dict(winSize = (10,10), # size of the search window at each pyramid level
                 maxLevel = 2, #  0, pyramids are not used (single level), if set to 1, two levels are used, and so on
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# same as above, with a bigger window size after blured images
lk_params_blur = lk_params.copy()
lk_params_blur.update(winSize = (50,50),
                      maxLevel = 3)

# params to farneback (dense flow detector)
downsize_flow = 3 # use only X part of center of the window to calculate flow. higher means smaller window
percent_flow = 0.2 # n percent of highest and lowest will be ignored to measure flow
rectify_flow_max = 2.0 # max value to multiply flow when image is blurier. when frame is blurier, flow tends to be underestimated
farneback_params = dict(pyr_scale = 0.5,
                        levels = 2,
                        winsize = 10,
                        iterations = 3,
                        poly_n = 5,
                        poly_sigma = 1.0,
                        flags = 0)

# same as above, with a bigger window size after blured images
farneback_params_blur = farneback_params.copy()
farneback_params_blur.update(levels = 4)


#### define variables to display output during runtime
window_size = 15 # time window to display the graph, in seconds

# Parameters for text display over the image
text_params = dict(org = (20,150),
                   fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale = 1.5,
                   color = (0,255,0),
                   thickness = 2)

text_params_sub = text_params.copy()
text_params_sub.update(org = (20,200))

# Parameters to resize images when displayed
resize_params = dict(dsize = (0,0),
                     fx = 0.5, fy = 0.5, 
                     interpolation = cv2.INTER_AREA)
