import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from time import time


def load_image(path):
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    return image


def disparity_map_using_NCC(img_left, img_right, disp_left, block_size):
    height, width = img_left.shape
    d_map = np.zeros(img_left.shape)
    max_disparity = int(disp_left.max() / 3)
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
    max_disparity = int(disp_left.max() / 3)
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


def calc_dMap_using_SSD(img_left, img_right, disp_left, win_size, pair_name):
    start_time = time()
    d_map = disparity_map_using_SSD(img_left, img_right, disp_left, win_size)
    print("(" + pair_name + ", SSD, win=" + str(win_size) + "), time= %.4f" % ((time() - start_time) / 60), "minutes")
    img_name = pair_name + ',SSD,win=' + str(win_size) + '.jpg'
    plt.imsave(img_name, d_map)
    img = load_image(img_name)
    cv.imwrite(img_name, img)


def calc_dMap_using_NCC(img_left, img_right, disp_left, win_size, pair_name):
    start_time = time()
    d_map = disparity_map_using_NCC(img_left, img_right, disp_left, win_size)
    print("(" + pair_name + ", NCC, win=" + str(win_size) + "), time= %.4f" % ((time() - start_time) / 60), "minutes")
    img_name = pair_name + ',NCC,win=' + str(win_size) + '.jpg'
    plt.imsave(img_name, d_map)
    img = load_image(img_name)
    cv.imwrite(img_name, img)


# def Art_Pair_SSD(img_left, img_right, disp_left):
#     """ -------------------SSD, window=3------------------- """
#     start_time0 = time()
#     d_map0 = disparity_map_using_SSD(img_left, img_right, disp_left, 3)
#     print("(Art_Pair, SSD, win=3), time= %.4f" % ((time() - start_time0) / 60), "minutes")
#     plt.imsave('Art_Pair,SSD,win=3.jpg', d_map0)
#     img0 = load_image('Art_Pair,SSD,win=3.jpg')
#     cv.imwrite('Art_Pair,SSD,win=3.jpg', img0)
#
#     """ -------------------SSD, window=9------------------- """
#     start_time1 = time()
#     d_map1 = disparity_map_using_SSD(img_left, img_right, disp_left, 9)
#     print("(Art_Pair, SSD, win=9), time= %.4f" % ((time() - start_time1) / 60), "minutes")
#     plt.imsave('Art_Pair,SSD,win=9.jpg', d_map1)
#     img1 = load_image('Art_Pair,SSD,win=9.jpg')
#     cv.imwrite('Art_Pair,SSD,win=9.jpg', img1)
#
#     """ -------------------SSD, window=15------------------- """
#     start_time2 = time()
#     d_map2 = disparity_map_using_SSD(img_left, img_right, disp_left, 15)
#     print("(Art_Pair, SSD, win=15), time= %.4f" % ((time() - start_time2) / 60), "minutes")
#     plt.imsave('Art_Pair,SSD,win=15.jpg', d_map2)
#     img2 = load_image('Art_Pair,SSD,win=15.jpg')
#     cv.imwrite('Art_Pair,SSD,win=15.jpg', img2)


# def Art_Pair_NCC(img_left, img_right, disp_left):
#     """ -------------------NCC, window=3------------------- """
#     start_time = time()
#     d_map = disparity_map_using_NCC(img_left, img_right, disp_left, 3)
#     print("(Art_Pair, NCC, win=3), time= %.4f" % ((time() - start_time) / 60), "minutes")
#     plt.imsave('Art_Pair,NCC,win=3.jpg', d_map)
#     img = load_image('Art_Pair,NCC,win=3.jpg')
#     cv.imwrite('Art_Pair,NCC,win=3.jpg', img)
#
#     """ -------------------NCC, window=9------------------- """
#     start_time = time()
#     d_map = disparity_map_using_SSD(img_left, img_right, disp_left, 9)
#     print("(Art_Pair, NCC, win=9), time= %.4f" % ((time() - start_time) / 60), "minutes")
#     plt.imsave('Art_Pair,NCC,win=9.jpg', d_map)
#     img = load_image('Art_Pair,NCC,win=9.jpg')
#     cv.imwrite('Art_Pair,NCC,win=9.jpg', img)
#
#     """ -------------------NCC, window=15------------------- """
#     start_time = time()
#     d_map = disparity_map_using_NCC(img_left, img_right, disp_left, 15)
#     print("(Art_Pair, NCC, win=15), time= %.4f" % ((time() - start_time) / 60), "minutes")
#     plt.imsave('Art_Pair,NCC,win=15.jpg', d_map)
#     img = load_image('Art_Pair,NCC,win=15.jpg')
#     cv.imwrite('Art_Pair,NCC,win=15.jpg', img)


def main():
    """----------------Pair Art----------------"""
    pairArt_img_left = load_image('Q2/Art/im_left.png')
    pairArt_img_right = load_image('Q2/Art/im_right.png')
    pairArt_disp_left = load_image('Q2/Art/disp_left.png')
    # calc_dMap_using_SSD(pairArt_img_left, pairArt_img_right, pairArt_disp_left, win_size=3, pair_name='Art_Pair')
    # calc_dMap_using_SSD(pairArt_img_left, pairArt_img_right, pairArt_disp_left, win_size=9, pair_name='Art_Pair')
    # calc_dMap_using_SSD(pairArt_img_left, pairArt_img_right, pairArt_disp_left, win_size=15, pair_name='Art_Pair')
    calc_dMap_using_NCC(pairArt_img_left, pairArt_img_right, pairArt_disp_left, win_size=3, pair_name='Art_Pair')

    """----------------Pair Dolls----------------"""
    pairDolls_img_left = load_image('Q2/Dolls/im_left.png')
    pairDolls_img_right = load_image('Q2/Dolls/im_right.png')
    pairDolls_disp_left = load_image('Q2/Dolls/disp_left.png')
    # calc_dMap_using_SSD(pairDolls_img_left, pairDolls_img_right, pairDolls_disp_left, win_size=3, pair_name='Dolls_Pair')
    # calc_dMap_using_SSD(pairDolls_img_left, pairDolls_img_right, pairDolls_disp_left, win_size=9, pair_name='Dolls_Pair')
    # calc_dMap_using_SSD(pairDolls_img_left, pairDolls_img_right, pairDolls_disp_left, win_size=15, pair_name='Dolls_Pair')
    calc_dMap_using_NCC(pairDolls_img_left, pairDolls_img_right, pairDolls_disp_left, win_size=3, pair_name='Dolls_Pair')


main()
