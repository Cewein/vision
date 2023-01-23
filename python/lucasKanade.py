import numpy as np
import scipy.signal as signal

from tqdm import tqdm
from python.error import endpoint_error, angular_error

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

    Ix1 = signal.convolve2d(I1, kernel_x, boundary='symm', mode='same')
    Ix2 = signal.convolve2d(I2, kernel_x, boundary='symm', mode='same')

    Ix = 0.25 * (Ix1 + Ix2)

    Iy1 = signal.convolve2d(I1, kernel_y, boundary='symm', mode='same')
    Iy2 = signal.convolve2d(I2, kernel_y, boundary='symm', mode='same')

    Iy = 0.25 * (Iy1 + Iy2)

    It1 = signal.convolve2d(I1, kernel_t, boundary='symm', mode='same')
    It2 = signal.convolve2d(I2, kernel_t, boundary='symm', mode='same')

    It = 0.25 * (It1 - It2)

    return Ix, Iy, It

def lucasKanade(I1,I2, winSize = 5):
    Ix, Iy, It = gradHorn(I1,I2)
    border = np.int_(winSize/2)
    width,heigth = np.array(I1).shape
    w = np.zeros((width,heigth,2))

    for x in range(border, width-border):
        for y in range(border, heigth-border):

            A = np.zeros((winSize*winSize,2))
            B = np.zeros((winSize*winSize))

            try:
                A[:, 0] = Ix[x-border:x+border+1, y-border:y+border+1].flatten()
                A[:, 1] = Iy[x-border:x+border+1, y-border:y+border+1].flatten()

                B = -It[x-border:x+border+1, y-border:y+border+1].flatten()
            except:
                A[:, 0] = Ix[x-border:x+border, y-border:y+border].flatten()
                A[:, 1] = Iy[x-border:x+border, y-border:y+border].flatten()

                B = -It[x-border:x+border, y-border:y+border].flatten()


            w[x, y] = np.linalg.lstsq(A, B, rcond=None)[0]
    return w

def lucasKanadeBartlett(I1,I2, winSize = 5):
    Ix, Iy, It = gradHorn(I1,I2)

    bartlettWindow = np.bartlett(winSize)
    border = np.int_(winSize/2)
    width,heigth = np.array(I1).shape
    w = np.zeros((width,heigth,2))

    for x in range(border, width-border):
        for y in range(border, heigth-border):

            A = np.zeros((winSize*winSize,2))
            B = np.zeros((winSize*winSize))

            try:
                A[:, 0] = (Ix[x-border:x+border+1, y-border:y+border+1]*bartlettWindow).flatten()
                A[:, 1] = (Iy[x-border:x+border+1, y-border:y+border+1]*bartlettWindow).flatten()

                B = -(It[x-border:x+border+1, y-border:y+border+1]*bartlettWindow).flatten()
            except:
                A[:, 0] = (Ix[x-border:x+border, y-border:y+border]*bartlettWindow).flatten()
                A[:, 1] = (Iy[x-border:x+border, y-border:y+border]*bartlettWindow).flatten()

                B = -(It[x-border:x+border, y-border:y+border]*bartlettWindow).flatten()


            w[x, y] = np.linalg.lstsq(A, B, rcond=None)[0]
    return w


def optimize_LK(func, I1, I2, wgt, min_win=1, max_win=21, step=1):
    '''returns the parameters for which the method give the best result,
    wrt. ground truth.'''

    best_epe = float('inf')
    epe = []
    best_ang = float('inf')
    ang = []
    best_norm = float('inf')
    norm = []
    best_win=[min_win for i in range(3)]
    wins = [i for i in range(min_win, max_win, step)] 
    
    for alpha in tqdm(wins):
        wobs = func(I1, I2, alpha)
        # register errors
        epe.append( endpoint_error(wobs, wgt)[0] )
        ang.append( angular_error(wobs, wgt)[0] )

        # select the best wins
        if epe[-1] < best_epe:
            best_epe = epe[-1]
            best_win[0] = alpha
        if ang[-1] < best_ang:
            best_ang = ang[-1]
            best_win[1] = alpha

    return wins, best_win, epe, best_epe, ang, best_ang, norm, best_norm

