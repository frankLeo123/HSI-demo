import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import os
# global variation
global H
global S
global I

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        # os.makedirs(path)
        os.mkdir(path)
        print("----new folder" + path)
    else:
        print("--this folder already exist")


def save_image(image,path):
    # hsv = cv2.cvtColor(pic1, cv2.COLOR_BGR2HSV)
    global H
    global S
    global I
    H, S, I = cv2.split(image)
    mkdir(path)
    cv2.imwrite(path + "img.png", pic1)
    cv2.imwrite(path + "hsv.png", image)
    # print("save ====" , type(H),len(H))
    cv2.imwrite(path + "h.png", H)
    cv2.imwrite(path + "s.png", S)
    cv2.imwrite(path + "i.png", I)
    print ("-- result in " + path)
    # cv2.imwrite("./debug/233.png",pic1)
    # cv2.waitKey(0)

def rgb2hsi(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    b, g, r = cv2.split(rgb_lwpImg)
    # 归一化到[0,1]
    b = b / 255.000
    g = g / 255.000
    r = r / 255.000
    hsi_lwpImg = rgb_lwpImg.copy()
    H, S, I = cv2.split(hsi_lwpImg)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j]))
            den = np.sqrt((r[i, j]-g[i, j])**2+(r[i, j]-b[i, j])*(g[i, j]-b[i, j]))

            theta = float(np.arccos(num/den))
            # print(theta)
            if den == 0:
                H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                # H = 2*3.14169265 - theta
                H = 2 * math.pi - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j]+g[i, j]+r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3 * min_RGB/sum

            # H = H/(2*3.14159265)
            H = H/(2 * math.pi)
            I = sum/3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_lwpImg[i, j, 0] = H*255
            hsi_lwpImg[i, j, 1] = S*255
            hsi_lwpImg[i, j, 2] = I*255
    return hsi_lwpImg

def plot(img):
    ar = H.flatten()
    plt.hist(ar, bins=256, normed=1, facecolor='r',alpha = 0.3, label = 'h',edgecolor='r', hold=1)
    ag = S.flatten()
    plt.hist(ag, bins=256, normed=1, facecolor='g',alpha = 0.5, label = 's', edgecolor='g', hold=1)
    ab = I.flatten()
    plt.hist(ab, bins=256, normed=1, facecolor='b',alpha = 0.4, label = 'i', edgecolor='b')
    plt.legend()
    plt.show()
    # plt.plot(x,ah)
    # plt.hist(ah, bins = 256,normed =1,facecolor = 'r',edgecolor = 'r',hold = 1)
    # plt.show()

def grads(img):
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            grad = img[i,j]



if __name__ == '__main__':
    pic1 = cv2.imread("./res/bad.png")
    path = "./debug/hsi/bad/"
    # cv2.imshow("img", pic1)
    # cv2.waitKey(0)
    hsi = rgb2hsi(pic1)
    save_image(hsi,path)
    # cv2.imshow("demo",H)
    plot(hsi)
    # cv2.waitKey(0)
    # split rgb to hsi

    print(os.name)
