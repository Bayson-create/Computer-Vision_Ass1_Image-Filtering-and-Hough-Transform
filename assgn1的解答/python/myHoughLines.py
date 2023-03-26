import numpy as np
import cv2

# def myHoughLines(img_hough, nLines):
#     # Perform non-maximum suppression to eliminate non-maximal values in the accumulator
#     kernel_size = 5
#     non_max_img = cv2.dilate(img_hough, np.ones((kernel_size,kernel_size),np.uint8))
#     non_max_img = np.where(img_hough == non_max_img, img_hough, 0)
    
#     # Find the coordinates of the nLines highest-scoring cells in the accumulator
#     coords = np.argpartition(-non_max_img.ravel(), nLines-1)[:nLines]
#     rhos = coords // img_hough.shape[1]
#     thetas = coords % img_hough.shape[1]
    
#     return rhos, thetas

from scipy.ndimage.filters import maximum_filter

def myHoughLines(img_hough, nLines):
    # Find local maxima in Hough transform
    local_maxima = maximum_filter(img_hough, size=5) == img_hough
    local_maxima = local_maxima.astype(np.float32)
    local_maxima *= img_hough  # Zero out non-maximum values

    # Extract coordinates of strongest peaks
    coordinates = np.argwhere(local_maxima > 0)
    values = local_maxima[local_maxima > 0]
    idx = np.argsort(values)[::-1][:nLines]
    coordinates = coordinates[idx]
    values = values[idx]

    # Convert coordinates to (rho, theta) format
    rhos = coordinates[:, 0] - img_hough.shape[0] // 2
    thetas = coordinates[:, 1] * np.pi / img_hough.shape[1]

    return rhos, thetas


import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load image and compute edges
    img = cv2.imread('../data/img01.jpg', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150)

    # Compute Hough transform
    img_hough = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=200)

    # Detect lines using myHoughLines function
    rhos, thetas = myHoughLines(img_hough, nLines=5)

    # Draw lines on image
    for rho, theta in zip(rhos, thetas):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho + img.shape[1] // 2
        y0 = b * rho + img.shape[0] // 2
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display image with detected lines
    cv2.imshow('Detected lines', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
