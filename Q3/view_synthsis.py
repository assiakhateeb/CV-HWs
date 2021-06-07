import numpy as np
import useful_python_code.sintel_io as sio
import cv2 as cv
import matplotlib.pyplot as plt


def back_projection(k, depth):
    fx, fy, u0, v0 = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    height = depth.shape[0]
    width = depth.shape[1]
    cam_points = np.zeros((height * width, 3))
    new_cam_points = np.zeros((height * width, 4))
    i = 0
    # Loop through each pixel in the image
    for v in range(height):
        for u in range(width):
            x = (u - u0) * depth[v, u] / fx
            y = (v - v0) * depth[v, u] / fy
            z = depth[v, u]
            cam_points[i] = [x, y, z]
            new_cam_points[i] = [x, y, z, 1]

            i += 1
    # print('new cam points = ', new_cam_points[1])
    return cam_points, new_cam_points  # 2d to 3d


def back_to_2d(intrinsic, depth, image):
    """ P = K[I|0]"""
    'x = M*N*X'
    Im = np.zeros(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    cam_points, new_cam_points = back_projection(intrinsic, depth)
    I_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    i = 0
    for v in range(height):
        for u in range(width):
            coor_2d = intrinsic.dot(I_0.dot(new_cam_points[i]))
            z = (coor_2d[2])
            x = int((coor_2d[0]) / z)
            y = int((coor_2d[1]) / z)

            if 0 <= y < height and 0 <= x < width:
                Im[y, x] = image[v, u]
            i += 1

    cv.imwrite('aa.png', Im)
    return Im


def main():
    im_path = "alley_2"
    image = cv.imread(im_path + '.png')
    d = sio.depth_read(im_path + '.dpt')  # depth_map = camera's Z axis
    intrinsic, extrinsic = sio.cam_read(im_path + '.cam')  # intrinsic matrix = K, N = extrinsic matrix
    """K is the camera intrinsics matrix"""
    k_inv = np.linalg.inv(intrinsic)
    Im = back_to_2d(intrinsic, d, image)


main()
