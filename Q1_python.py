# imports packages
print("loading packages ...")
import os
import cv2 as cv
import time
import math
import numpy as np
from matplotlib import pyplot as plt
import random as random
print("packages loaded!")

# global variables
number_of_points = 1


# Methods
def Algebraic_Distance(pts1, pts2, F1):
    algebraic_distance = 0
    for i in range(len(pts1)):
        algebraic_distance += (pts2[i].T.dot(F1)).dot(pts1[i])
    algebraic_distance = algebraic_distance / (len(pts1))
    return algebraic_distance


def Epipolar_Distance(pts1, pts2, lines1, lines2, F1):
    epipolar_distance = 0
    for i in range(len(pts1)):
        x1 = (pts2[i].T.dot(F1)).dot(pts1[i]) / (math.sqrt(lines1[i][0] ** 2 + lines1[i][1] ** 2))
        x2 = (pts1[i].T.dot(F1.T)).dot(pts2[i]) / (math.sqrt(lines2[i][0] ** 2 + lines2[i][1] ** 2))
        epipolar_distance += ((x1 ** 2) + (x2 ** 2))
    return epipolar_distance


def drawlines(img1, img2, lines, pts1, pts2):
    """ drawlines:
    to draw all line in lines
    """
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        random_number1 = random.randint(0, 255)
        random_number2 = random.randint(0, 255)
        random_number3 = random.randint(0, 255)
        color = (random_number1, random_number2, random_number3)
        x0, y0 = map(int, [0, - r[2] / r[1]])
        x1, y1 = map(int, [c, - (r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 4)
        img1 = cv.circle(img1, tuple(pt1), 15, color, 3)
        img2 = cv.circle(img2, tuple(pt2), 15, color, 3)
    return img1, img2


def Q1(im1, im2, pts_left, pts_right, method):
    im1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    if method == 1:
        print("7 Point Algorithm: ")
        F1, mask1 = cv.findFundamentalMat(pts_left, pts_right, cv.FM_7POINT)
    if method == 2:
        print("8 Point Algorithm: ")
        F1, mask1 = cv.findFundamentalMat(pts_left, pts_right, cv.FM_8POINT)

    lines1, lines2, pts1, pts2 = list(), list(), list(), list()

    for point1 in pts_right:
        np_temp = np.array(point1)
        vec = np.append(np_temp, 1)
        pts2.append(vec)
        value = (F1.T).dot(vec)
        lines1.append(value)

    for point2 in pts_left:
        vec = np.array(point2)
        vec = np.append(vec, 1)
        pts1.append(vec)
        value = (F1).dot(vec)
        lines2.append(value)

    # change the type to numpy
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    lines1 = np.array(lines1)
    lines2 = np.array(lines2)

    # Draw lines and points
    left_img, junk = drawlines(im1, im2, lines1, pts_left, pts_right)
    right_img, junk = drawlines(im2, im1, lines2, pts_right, pts_left)
    plt.subplot(121), plt.imshow(left_img)
    plt.subplot(122), plt.imshow(right_img)
    plt.show()

    # Calculating the distance's
    print("The Algebraic Distance is :", Algebraic_Distance(pts1, pts2, F1))
    print("The (Symmetric) Epipolar Distance is :",
          Epipolar_Distance(pts1, pts2, lines1, lines2, F1) / number_of_points)


def epipolar_geometry_UI(image_name1, image_name2):
    global number_of_points
    im1 = cv.imread(image_name1)
    im2 = cv.imread(image_name2)
    cv.imshow("Left image:", im1)
    cv.imshow("Right image:", im2)
    cv.waitKey(0)
    if image_name1 == "im_courtroom_00086_left.jpg" and image_name2 == "im_courtroom_00089_right.jpg":
        pts_left = np.array(
            [[207, 40], [280, 130], [342, 228], [406, 438], [32, 308], [617, 387],
             [696, 474], [838, 212], [725, 62], [427, 46]]).astype(np.float32)
        pts_right = np.array(
            [[265, 17], [313, 90], [360, 171], [537, 282], [287, 224], [626, 255],
             [661, 304], [732, 141], [653, 36], [405, 13]]).astype(np.float32)
        number_of_points = 10
    elif image_name1 == "im_family_00084_left.jpg" and image_name2 == "im_family_00100_right.jpg":
        pts_left = np.array(
            [[857, 210], [869, 314], [234, 328], [321, 288], [124, 51], [57, 253],
             [470, 261], [218, 195]]).astype(np.float32)
        pts_right = np.array(
            [[806, 211], [862, 417], [126, 332], [324, 294], [579, 32], [534, 235],
             [736, 270], [751, 185]]).astype(np.float32)
        number_of_points = 8
    else:
        print("There is no saved image with this name, ERROR!")
        return

    Q1(im1, im2, pts_left, pts_right, 1)
    cv.destroyAllWindows()
    print("#######")
    Q1(im1, im2, pts_left, pts_right, 2)
    cv.destroyAllWindows()


if __name__ == "__main__":
    epipolar_geometry_UI("im_courtroom_00086_left.jpg", "im_courtroom_00089_right.jpg")
    epipolar_geometry_UI("im_family_00084_left.jpg", "im_family_00100_right.jpg")
