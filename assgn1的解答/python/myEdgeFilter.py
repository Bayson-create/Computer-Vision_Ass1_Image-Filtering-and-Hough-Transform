import numpy as np
from scipy import signal    # For signal.gaussian function

import matplotlib.pyplot as plt #new added

from myImageFilter import myImageFilter

import cv2

def myEdgeFilter(img0, sigma):
    # YOUR CODE HERE
    # kernel to be used for edge detection
    # image = signal.ndimage.gaussian_filter(img0, sigma)
    # print(image)
    hsize = int(2 * np.ceil(3 * sigma) + 1)
        
    print(hsize)
    
    kernel = gaussian_kernel(l=3, sig=sigma)
    print(kernel)
    
    image = myImageFilter(img0, kernel)
    # image = img0
    
    # sobelx = cv2.Sobel(img0, -1, 1, 0)
    # sobely = cv2.Sobel(img0, -1, 0, 1)
    
    sobelx = myImageFilter(image, [[1,0,-1], [2,0,-2], [1,0,-1]])
    sobely = myImageFilter(image, [[1,2,1], [0,0,0], [-1,-2,-1]])

    # image = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    
    #计算梯度幅值和方向
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    orientation = np.arctan2(sobely, sobelx) * 180 / np.pi
    
    #对梯度方向进行量化
    quantized_orientation = np.zeros_like(orientation)
    quantized_orientation[np.where((orientation >= -22.5) & (orientation < 22.5))] = 0
    quantized_orientation[np.where((orientation >= 22.5) & (orientation < 67.5))] = 45
    quantized_orientation[np.where((orientation >= 67.5) & (orientation < 112.5))] = 90
    quantized_orientation[np.where((orientation >= 112.5) & (orientation < 157.5))] = 135
    
    #对梯度幅值进行NMS
    nms_magnitude = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            direction = quantized_orientation[i, j]
            if direction == 0:
                if magnitude[i, j] >= magnitude[i, j-1] and magnitude[i, j] >= magnitude[i, j+1]:
                    nms_magnitude[i, j] = magnitude[i, j]
            elif direction == 45:
                if magnitude[i, j] >= magnitude[i-1, j+1] and magnitude[i, j] >= magnitude[i+1, j-1]:
                    nms_magnitude[i, j] = magnitude[i, j]
            elif direction == 90:
                if magnitude[i, j] >= magnitude[i-1, j] and magnitude[i, j] >= magnitude[i+1, j]:
                    nms_magnitude[i, j] = magnitude[i, j]
            elif direction == 135:
                if magnitude[i, j] >= magnitude[i-1, j-1] and magnitude[i, j] >= magnitude[i+1, j+1]:
                    nms_magnitude[i, j] = magnitude[i, j]                             
                    
    #应用阈值
    thresh = 50
    edges = np.zeros_like(nms_magnitude)
    edges[np.where(nms_magnitude > thresh)] = 255
    
    #显示结果
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Edges', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    # # kernel = np.ones((4, 4), np.uint8)
    # image = cv2.dilate(image,kernel,iterations=1)
    # image = py_cpu_nms(image, 0.7)
    
    # return image
    return edges
    
    print(sobelx)
    print(sobely)
    
    # sobel-x方向
    # sobel_X = cv2.convertScaleAbs(sobelx)
    # sobel-y方向
    # sobel_Y = cv2.convertScaleAbs(sobely)
    # sobel-xy方向
    # scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)

    # image_edge1 = myImageFilter(img0, h=kernel)
    # cv2.imwrite('edge_detection1.jpg', image_edge1)

    # image_edge2 = myImageFilter(img0,
    #                         h=np.array(sigma))
    # cv2.imwrite('edge_detection2.jpg', image_edge2)
    
def gaussian_kernel(l=3, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def py_cpu_nms(dets, thresh): 
    #dets某个类的框，x1、y1、x2、y2、以及置信度score
    #eg:dets为[[x1,y1,x2,y2,score],[x1,y1,y2,score]……]]
    # thresh是IoU的阈值     
    x1 = dets[:, 0] 
    y1 = dets[:, 1]
    x2 = dets[:, 2] 
    y2 = dets[:, 3] 
    scores = dets[:, 4] 
    #每一个检测框的面积 
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    #按照score置信度降序排序 
    order = scores.argsort()[::-1] 
    keep = [] #保留的结果框集合 
    while order.size > 0: 
        i = order[0] 
        keep.append(i) #保留该类剩余box中得分最高的一个 
        #得到相交区域,左上及右下 
        xx1 = np.maximum(x1[i], x1[order[1:]]) 
        yy1 = np.maximum(y1[i], y1[order[1:]]) 
        xx2 = np.minimum(x2[i], x2[order[1:]]) 
        yy2 = np.minimum(y2[i], y2[order[1:]]) 
        #计算相交的面积,不重叠时面积为0 
        w = np.maximum(0.0, xx2 - xx1 + 1) 
        h = np.maximum(0.0, yy2 - yy1 + 1) 
        inter = w * h 
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积） 
        ovr = inter / (areas[i] + areas[order[1:]] - inter) 
       #保留IoU小于阈值的box 
        inds = np.where(ovr <= thresh)[0] 
        order = order[inds + 1] #因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位 
    return keep

if __name__ == '__main__':
    f = cv2.imread('../data/img01.jpg', cv2.IMREAD_GRAYSCALE)
    image = f.copy()
    if (image.ndim == 3):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    plt.subplot(121)
    plt.imshow(f, vmin=0, vmax=255, cmap='gray')
    image = myEdgeFilter(image, 2)
    # image = signal.convolve(image, [[1, 0, -1], [1, 0, -1], [1, 0, -1]], mode="valid")
    print(image)

    plt.subplot(122)
    plt.imshow(image, vmin=0, vmax=255, cmap='gray')
    plt.show()