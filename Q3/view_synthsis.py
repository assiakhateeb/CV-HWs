import numpy as np
import useful_python_code.sintel_io as sio
import cv2 as cv
import copy as cp
import os
from time import time


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
    return cam_points, new_cam_points  # 2d to 3d


def back_to_2d(intrinsic, depth, image, extrinsic):
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
            # coor_2d = intrinsic.dot(I_0.dot(new_cam_points[i]))
            coor_2d = intrinsic.dot(extrinsic.dot(new_cam_points[i]))
            z = (coor_2d[2])
            x = int((coor_2d[0]) / z)
            y = int((coor_2d[1]) / z)

            if 0 <= y < height and 0 <= x < width:
                Im[y, x] = image[v, u]
            i += 1

    # cv.imwrite('aa.png', Im)
    return Im


def rotation_mat(theta_x, theta_y, theta_z):
    rotate_x = np.matrix([[1, 0, 0, 0],
                          [0, np.cos(theta_x), -np.sin(theta_x), 0],
                          [0, np.sin(theta_x), np.cos(theta_x), 0],
                          [0, 0, 0, 1]])

    rotate_y = np.matrix([[np.cos(theta_y), 0, np.sin(theta_y), 0],
                          [0, 1, 0, 0],
                          [-np.sin(theta_y), 0, np.cos(theta_y), 0],
                          [0, 0, 0, 1]])

    rotate_z = np.matrix([[np.cos(theta_z), -np.sin(theta_z), 0, 0],
                          [np.sin(theta_z), np.cos(theta_z), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

    return rotate_x, rotate_y, rotate_z


def new_pose(rotation=None, translation=None):
    if rotation is None:
        rotation = np.array([0, 0, 0])
    if translation is None:
        translation = np.array([0, 0, 0])

    ext = [[1, 0, 0, translation[0]], [0, 1, 0, translation[1]], [0, 0, 1, translation[2]], [0, 0, 0, 1]]
    ext = np.array(ext)
    x, y, z = rotation_mat(rotation[0] * np.pi / 180, rotation[1] * np.pi / 180, rotation[2] * np.pi / 180)
    pose = x.dot(y.dot(z.dot(ext)))
    return pose


def make_view(im_path, image, folder):
    start_time = time()
    depth = sio.depth_read(im_path + '.dpt')  # depth_map = camera's Z axis
    intrinsic, extrinsic = sio.cam_read(im_path + '.cam')  # intrinsic matrix = K, N = extrinsic matrix
    """K is the camera intrinsics matrix"""
    k_inv = np.linalg.inv(intrinsic)
    j = 1
    rotation = [0, 0, 0]
    translation = [0, 0, 0]
    r = np.array([0., 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.])
    r_frames = [r, -r, -r, r]
    t = np.array([0., 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.])
    t_frames = [t, -t, -t, t]
    for rotationOrTranslation in ['rotation', 'translation']:
        for axis in [0, 1, 2]:
            frames = r_frames
            if rotationOrTranslation == 'translation':
                frames = t_frames
            for frame in frames:
                for i in frame:
                    if rotationOrTranslation == 'translation':
                        translation[axis] += i
                    else:
                        rotation[axis] += i
                    pose = new_pose(cp.deepcopy(rotation), cp.deepcopy(translation))
                    pose = np.array([pose[0], pose[1], pose[2]])
                    Im1 = back_to_2d(intrinsic, depth, image, pose)
                    cv.imwrite(folder + '/' + str(j) + '.png', Im1)
                    j += 1
    end_time = (time() - start_time) / 60
    print(im_path, "time=%.4f" % end_time, "minutes,")


def main():
    folder = 'view-synthesis1'
    os.makedirs(folder)
    im_path = "alley_2"
    alley_image = cv.imread(im_path + '.png')
    make_view(im_path, alley_image, folder)

    folder = 'view-synthesis2'
    os.makedirs(folder)
    im_path = "ambush_6"
    ambush_image = cv.imread(im_path + '.png')
    make_view(im_path, ambush_image, folder)


main()
