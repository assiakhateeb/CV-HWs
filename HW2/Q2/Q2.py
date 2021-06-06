import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from time import time
import sys
import os


def load_image(path):
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return image


def disparity_map_using_NCC(img_left, img_right, disp_left, block_size):
    height, width = img_left.shape
    d_map = np.zeros(img_left.shape)
    max_disparity = int(disp_left.max())
    shift = 0
    for row in range(height - block_size + 1):
        for col in range(width - block_size + 1):
            if col - block_size // 2 >= 0 and col + block_size // 2 + 1 < width and row - block_size // 2 >= 0 and row + block_size // 2 + 1 < height:
                best_dist = float('-inf')
                left_block = np.array(img_left[row:row + block_size, col:col + block_size])
                for i in range(col - max_disparity, col + 1):
                    if i - block_size // 2 >= 0 and block_size // 2 + i < width:
                        right_block = np.array(img_right[row:row + block_size, i:i + block_size])
                        """flatten function Return a copy of the array collapsed into one dimension."""
                        """np.correlate return the cross-correlation of two 1-dimensional sequences."""
                        NCC_val = np.correlate(left_block.flatten() / np.linalg.norm(left_block),
                                               right_block.flatten() / np.linalg.norm(right_block))
                        if NCC_val > best_dist:
                            best_dist = NCC_val
                            shift = i
                d_map[row, col] = abs(shift - col)
    return d_map


def disparity_map_using_SSD(img_left, img_right, disp_left, block_size):
    height, width = img_left.shape
    d_map = np.zeros(img_left.shape)
    max_disparity = int(disp_left.max())
    shift = 0
    for row in range(height - block_size + 1):
        for col in range(width - block_size + 1):
            if col - block_size // 2 >= 0 and col + block_size // 2 + 1 < width and row - block_size // 2 >= 0 and row + block_size // 2 + 1 < height:
                best_dist = float('inf')
                left_block = np.array(img_left[row:row + block_size, col:col + block_size])
                for i in range(col - max_disparity, col + 1):
                    if i - block_size // 2 >= 0 and block_size // 2 + i < width:
                        right_block = np.array(img_right[row:row + block_size, i:i + block_size])
                        ssd = np.sum((left_block - right_block) ** 2)
                        if ssd < best_dist:
                            best_dist = ssd
                            shift = i
                d_map[row, col] = abs(shift - col)
    return d_map


def bad05_bad4(mapA, mapB):
    diff = np.abs(mapA - mapB)
    elements_num = (diff >= 0).sum()
    """Bad05 = percentage of disparities whose error is above 0.5 """
    bad5 = (diff > 0.5).sum()
    """Bad4 = percentage of disparities whose error is above 4 """
    bad4 = (diff > 4).sum()
    percent05 = (bad5 / elements_num) * 100
    percent4 = ((bad4 / elements_num) * 100)
    return percent05, percent4


def utilts_1(img_left, img_right, disp_left, win_size, pair_name):
    start_time = time()
    d_map = disparity_map_using_SSD(img_left, img_right, disp_left, win_size)
    end_time = (time() - start_time) / 60
    """avgErr = mean of absolute differences in pixels"""
    avgErr = np.mean(np.absolute(d_map - disp_left))
    """medErr = median of absolute differences in pixels"""
    medErr = np.median(np.absolute(d_map - disp_left))
    bad05, bad4 = bad05_bad4(d_map, disp_left)
    print("(" + pair_name + ", SSD, win=" + str(win_size) + "), time=%.4f" % end_time, "minutes,",
          "AvgErr=%.4f," % avgErr, "medErr=%.4f," % medErr, "Bad05=%.4f," % bad05, "%", "Bad4=%.4f" % bad4, "%")
    img_name = 'results_for_SSD/' + pair_name + ' ,SSD,win=' + str(win_size) + str(
        " ,AvgErr=%.4f" % avgErr) + " ,MedErr=%.4f" % medErr + " ,Bad05=%.4f" % bad05 + "%" + " ,Bad4=%.4f" % bad4 + "%" + '.jpg'
    plt.imsave(img_name, d_map)
    img = load_image(img_name)
    cv.imwrite(img_name, img)


def utilts_2(img_left, img_right, disp_left, win_size, pair_name):
    start_time = time()
    d_map = disparity_map_using_NCC(img_left, img_right, disp_left, win_size)
    end_time = (time() - start_time) / 60
    avgErr = np.mean(np.absolute(d_map - disp_left))
    medErr = np.median(np.absolute(d_map - disp_left))
    bad05, bad4 = bad05_bad4(d_map, disp_left)
    print("(" + pair_name + ", NCC, win=" + str(win_size) + "), time=%.4f" % end_time, "minutes,",
          "AvgErr=%.4f," % avgErr, "medErr=%.4f," % medErr, "Bad05=%.4f," % bad05, "%", "Bad4=%.4f" % bad4, "%")
    img_name = 'results_for_NCC/' + pair_name + ' ,NCC,win=' + str(win_size) + str(
        " ,AvgErr=%.4f" % avgErr) + " ,MedErr=%.4f" % medErr + " ,Bad05=%.4f" % bad05 + "%" + " ,Bad4=%.4f" % bad4 + "%" + '.jpg'
    plt.imsave(img_name, d_map)
    img = load_image(img_name)
    cv.imwrite(img_name, img)


def main():
    """----------------Pair Art----------------"""
    pairArt_img_left = load_image('Q2/Art/im_left.png')
    pairArt_img_right = load_image('Q2/Art/im_right.png')
    pairArt_disp_left = load_image('Q2/Art/disp_left.png')
    pairArt_disp_left = np.array(pairArt_disp_left) / 3
    """----------------Pair Dolls----------------"""
    pairDolls_img_left = load_image('Q2/Dolls/im_left.png')
    pairDolls_img_right = load_image('Q2/Dolls/im_right.png')
    pairDolls_disp_left = load_image('Q2/Dolls/disp_left.png')
    pairDolls_disp_left = np.array(pairDolls_disp_left) / 3
    """----------------Pair Moebius----------------"""
    pairMoebius_img_left = load_image('Q2/Moebius/im_left.png')
    pairMoebius_img_right = load_image('Q2/Moebius/im_right.png')
    pairMoebius_disp_left = load_image('Q2/Moebius/disp_left.png')
    pairMoebius_disp_left = np.array(pairMoebius_disp_left) / 3
    window = [3, 9, 15]   

    """--------------First Calc disparity map using SSD for all image pairs--------------"""
    folder = 'results_for_SSD'
    os.makedirs(folder)
    sys.stdout = open("results_for_SSD/results_for_SSD.txt", "w")
    for i in window:
        utilts_1(pairArt_img_left, pairArt_img_right, pairArt_disp_left, win_size=i, pair_name='Art_Pair')
        utilts_1(pairDolls_img_left, pairDolls_img_right, pairDolls_disp_left, win_size=i, pair_name='Dolls_Pair')
        utilts_1(pairMoebius_img_left, pairMoebius_img_right, pairMoebius_disp_left, win_size=i,
                 pair_name='Moebius_Pair')
    sys.stdout.close()

    """--------------Second Calc disparity map using NCC for all image pairs--------------"""
    folder = 'results_for_NCC'
    os.makedirs(folder)
    sys.stdout = open("results_for_NCC/results_for_NCC.txt", "w")
    for i in window:
        utilts_2(pairArt_img_left, pairArt_img_right, pairArt_disp_left, win_size=i, pair_name='Art_Pair')
        utilts_2(pairDolls_img_left, pairDolls_img_right, pairDolls_disp_left, win_size=i,
                 pair_name='Dolls_Pair')
        utilts_2(pairMoebius_img_left, pairMoebius_img_right, pairMoebius_disp_left, win_size=i,
                 pair_name='Moebius_Pair')
    sys.stdout.close()


main()
