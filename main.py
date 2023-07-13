import cv2
import numpy as np

emptytable = cv2.imread('base.jpg')
fullytable = cv2.imread('input.jpg')


# we need to find all squares in the image
def get_difer(img1, img2):
    global threshold_value
    # gray scale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(img1, img2)

    # apply th
    _, output = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    return output


def getContours(img):
    global maxSize
    countours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    squares = []
    for cnt in countours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > maxSize:
            squares.append(approx)
    return squares


cv2.namedWindow('Image with Squares')
threshold_value = 0
maxSize = 748


def update_threshold_value(value):
    global threshold_value
    threshold_value = value


def update_max_size_value(value):
    global maxSize
    maxSize = value


# Create a trackbar to control the threshold value
cv2.createTrackbar('Threshold', 'Image with Squares',
                   threshold_value, 255, update_threshold_value)
cv2.createTrackbar('MaxSize', 'Image with Squares',
                   maxSize, 10000, update_max_size_value)

while True:
    fullytable = cv2.imread('input.jpg')
    diff_img = get_difer(emptytable, fullytable)
    squares = getContours(diff_img)
    for square in squares:
        cv2.drawContours(fullytable, [square], -1, (0, 255, 0), 3)
    # get the in the image
    try:
        img_crop = fullytable[squares[0][0][0][1]:squares[0][2][0][1],
                              squares[0][0][0][0]:squares[0][2][0][0]]
        if len(squares) > 0:
            cv2.imshow('crop', img_crop)
    except:
        pass
    cv2.imshow('diff', diff_img)
    cv2.imshow('fullytable', fullytable)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
