import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# image = cv2.imread('test_image.jpg')
# lane_img = np.copy(image)
# PREDICT_DIST = 3/5
# WIDTH = lane_img.shape[1]
# HEIGHT = lane_img.shape[0]

cap = cv2.VideoCapture("test3.mp4")
PREDICT_DIST = 3/5
# START =
WIDTH = int(cap.get(3))
print(WIDTH)
HEIGHT = int(cap.get(4))
print(HEIGHT)

# creates an outline of the image based on gradient


def canny(image):
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(grayscale_img, (5, 5), 0)
    canny_img = cv2.Canny(blur_img, 50, 150)
    return canny_img

# creates a mask for the region of interest and crops the image


def region_of_interest(image):
    # height = image.shape[0]
    polygons = np.array([
        [(200, HEIGHT), (1100, HEIGHT), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def make_coordinates(image, line_params):
    slope, y_intercept = line_params
    # y1 = image.shape[0]
    y1 = HEIGHT
    y2 = int(y1 * PREDICT_DIST)
    x1 = int((y1 - y_intercept)/slope)
    x2 = int((y2 - y_intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope, y_intercept = params[0], params[1]
        if slope < 0:
            left_fit.append((slope, y_intercept))
        else:
            right_fit.append((slope, y_intercept))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)
    return np.array([left_line, right_line])


def display_lines(image, lines):
    line_img = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_img


def heading(lane_lines):
    _, _, left_x2, _ = lane_lines[0]
    _, _, right_x2, _ = lane_lines[1]
    midpoint = int(WIDTH / 2)
    x_offset = ((left_x2 + right_x2) / 2 - midpoint)
    y_offset = int(HEIGHT * PREDICT_DIST)

    # adjust steering angle
    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180 / math.pi)
    return angle_to_mid_deg


def display_heading(image, angle):
    heading_img = np.zeros_like(image)
    print("Steering should turn by " + str(angle) + " degrees.")
    angle_radian = angle / 180 * math.pi
    x1 = int(WIDTH / 2)
    y1 = HEIGHT
    x2 = int(x1 + HEIGHT * PREDICT_DIST * math.tan(angle_radian))
    y2 = int(HEIGHT * PREDICT_DIST)

    cv2.line(heading_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return heading_img


# canny_img = canny(lane_img)
# cropped_img = region_of_interest(canny_img)
# lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100,
#                         np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_img, lines)
# line_img = display_lines(lane_img, averaged_lines)
# heading_img = display_heading(lane_img, heading(averaged_lines))

# lines_combo_img = cv2.addWeighted(line_img, 1, heading_img, 1, 1)
# combo_img = cv2.addWeighted(lane_img, 0.8, lines_combo_img, 1, 1)

# cv2.imshow("result", combo_img)
# cv2.waitKey(0)


while(cap.isOpened()):
    _, frame = cap.read()
    canny_img = canny(frame)

    cropped_img = region_of_interest(canny_img)

    lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100,
                            np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_img = display_lines(frame, averaged_lines)
    heading_img = display_heading(frame, heading(averaged_lines))

    lines_combo_img = cv2.addWeighted(line_img, 1, heading_img, 1, 1)
    combo_img = cv2.addWeighted(frame, 0.8, lines_combo_img, 1, 1)

    cv2.imshow("result", combo_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# plt.imshow(canny_img)
# plt.show()
