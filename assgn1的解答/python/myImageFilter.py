import numpy as np

import cv2 #new added
import matplotlib.pyplot as plt #new added
from scipy import signal    # For signal.gaussian function

def myImageFilter(img0, h):
    # YOUR CODE HERE    
    n = np.size(h, 0)
    m = np.size(h, 1)
    
    print(img0)
    
    conv_heigh = img0.shape[0]
    conv_width = img0.shape[1]
    
    image_heigh = conv_heigh + n - 1
    image_width = conv_width + m - 1
    
    #add zero padding to the input image img0
    image_padded = np.zeros((image_heigh, image_width))
    output = np.zeros((conv_heigh, conv_width))
    print(np.size(image_padded, 0))
    print(conv_heigh)
    image_padded[n-1:image_heigh, m-1:image_width] = img0
    for x in range(n-1):
        for y in range(m-1, image_width):
            image_padded[x][y] = img0[0][y-m+1]
    for y in range(m-1):
        for x in range(n-1, image_heigh):
            image_padded[x][y] = img0[x-n+1][0]
    print(image_padded)
    
    for x in range(conv_heigh):
        for y in range(conv_width):
            temp = (image_padded[x:x+n, y:y+m] * h).sum()
            if (temp < 0): temp = 0
            elif (temp > 255): temp = 255
            output[x][y] = temp
            
    print(output)
    return output

#test
if __name__ == '__main__':
    f = cv2.imread('../data/img01.jpg', cv2.IMREAD_GRAYSCALE)
    image = f.copy()
    if (image.ndim == 3):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    plt.subplot(121)
    plt.imshow(f, vmin=0, vmax=255, cmap='gray')
    image = myImageFilter(image, [[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    # image = signal.convolve(image, [[1, 0, -1], [1, 0, -1], [1, 0, -1]], mode="valid")
    print(image)

    plt.subplot(122)
    plt.imshow(image, vmin=0, vmax=255, cmap='gray')
    plt.show()