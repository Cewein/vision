import python.middlebury
import numpy as np
import python.middlebury
import matplotlib.pyplot as plt
import scipy as sp

from PIL import Image

def gradHorn(I1,I2):

    kernel_y = -np.array([
        [0,0,0],
        [0,-1,-1],
        [0,1,1]
    ])

    kernel_x = -np.array([
        [0,0,0],
        [0,-1,1],
        [0,-1,1]
    ])

    kernel_t = -np.array([
        [0,0,0],
        [0,1,1],
        [0,1,1]
    ])

    Ix1 = sp.signal.convolve2d(I1, kernel_x, boundary='symm', mode='same')
    Ix2 = sp.signal.convolve2d(I2, kernel_x, boundary='symm', mode='same')

    Ix = 0.25 * (Ix1 + Ix2)

    Iy1 = sp.signal.convolve2d(I1, kernel_y, boundary='symm', mode='same')
    Iy2 = sp.signal.convolve2d(I2, kernel_y, boundary='symm', mode='same')

    Iy = 0.25 * (Iy1 + Iy2)

    It1 = sp.signal.convolve2d(I1, kernel_t, boundary='symm', mode='same')
    It2 = sp.signal.convolve2d(I2, kernel_t, boundary='symm', mode='same')

    It = 0.25 * (It1 - It2)

    return Ix, Iy, It

# I1 = plt.imread("data/nasa/nasa9.png")
# I2 = plt.imread("data/nasa/nasa10.png")

# I1 = plt.imread("data/yosemite/yos9.png")
# I2 = plt.imread("data/yosemite/yos10.png")

I1 = Image.open("data/rubberwhale/frame10.png").convert('L')
I2 = Image.open("data/rubberwhale/frame11.png").convert('L')


def lucasKanade(I1,I2, winSize = 5)
    Ix, Iy, It = gradHorn(I1,I2)
    
    border = np.int_(winSize/2)
    
    width,heigth = np.array(I1).shape
    
    nb = 0
    
    w = np.zeros((width,heigth,2))
    
    for x in range(border, width-border):
        for y in range(border, heigth-border):
            
            A = np.zeros((size*size,2))
            B = np.zeros((size*size))
            
            for i in range(-border,border+1):
                for j in range(-border,border+1):
                    B[i + winSize*j] = -It[x+i][y+j]
                    A[i + winSize*j][0] = Ix[x+i][y+j]
                    A[i + winSize*j][1] = Iy[x+i][y+j]
        
        w[x][y] = np.linalg.inv(A.T@A)@(A.T@B)

# %%


img = python.middlebury.computeColor(w)

plt.imshow(img)
plt.show()


