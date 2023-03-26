import numpy as np
from myEdgeFilter import myEdgeFilter

def myHoughTransform(img_threshold, rho_res, theta_res):
    height, width = img_threshold.shape
    max_rho = np.ceil(np.sqrt(height**2 + width**2)).astype(int)
    num_theta = int(np.round(np.pi / theta_res))
    rho_scale = np.arange(0, max_rho + 1, rho_res)
    theta_scale = np.linspace(0, np.pi, num_theta, endpoint=False)

    # initialize Hough transform accumulator
    img_hough = np.zeros((len(rho_scale), len(theta_scale)), dtype=int)

    # find edge points above threshold
    edge_points = np.nonzero(img_threshold > 0)
    y_points, x_points = edge_points

    # iterate through edge points and vote in Hough transform accumulator
    for i in range(len(x_points)):
        x = x_points[i]
        y = y_points[i]
        for j in range(len(theta_scale)):
            rho = int(x * np.cos(theta_scale[j]) + y * np.sin(theta_scale[j]))
            if rho >= 0 and rho < len(rho_scale):
                img_hough[rho, j] += 1

    return img_hough, rho_scale, theta_scale

import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # read image and apply Canny edge detection
    f = cv2.imread('../data/img03.jpg', cv2.IMREAD_GRAYSCALE)
    img = f.copy()
    if (img.ndim == 3):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plt.subplot(121)
    plt.imshow(f, vmin=0, vmax=255, cmap='gray')
    # edges = myEdgeFilter(img, 2)
    edges = cv2.Canny(img, 50, 150)
    rhoRes    = 2
    thetaRes  = np.pi / 90
    num_theta = int(np.round(np.pi / thetaRes))
    height = img.shape[0]
    width = img.shape[1]
    max_rho = np.ceil(np.sqrt(height**2 + width**2)).astype(int)
    # apply Hough transform and display output
    hough, rho_scale, theta_scale = myHoughTransform(edges, 1, np.pi/180)
    plt.subplot(122)
    plt.imshow(hough, vmin=0, vmax=255, cmap='gray')
    plt.xlabel('theta (radians)')
    plt.ylabel('rho (pixels)')
    plt.title('Hough Transform')
    plt.show()