import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in and grayscale the image
image = mpimg.imread('exit-ramp.jpg')
ysize = image.shape[0]
xsize = image.shape[1]
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

left_bottom = [80, 539]
right_bottom = [880, 539]
apex = [470, 280]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

rho = 1
theta = np.pi / 180
threshold = 1
min_line_length = 5
max_line_gap = 7
line_image = np.copy(image)*0

lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1,y1),(x2,y2),(255,0,0),7)

line_image[~region_thresholds] = [0,0,0]

color_edges = np.dstack((edges, edges, edges))
plt.imshow(color_edges)
plt.show()

combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
# plt.imshow(combo)
# plt.show()
