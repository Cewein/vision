import numpy as np
import python.middlebury
import scipy.signal as signal
from PIL import Image
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

    Ix1 =  signal.convolve2d(I1, kernel_x, boundary='symm', mode='same')
    Ix2 =  signal.convolve2d(I2, kernel_x, boundary='symm', mode='same')

    Ix = 0.25 * (Ix1 + Ix2)

    Iy1 =  signal.convolve2d(I1, kernel_y, boundary='symm', mode='same')
    Iy2 =  signal.convolve2d(I2, kernel_y, boundary='symm', mode='same')

    Iy = 0.25 * (Iy1 + Iy2)

    It1 =  signal.convolve2d(I1, kernel_t, boundary='symm', mode='same')
    It2 =  signal.convolve2d(I2, kernel_t, boundary='symm', mode='same')

    It = 0.25 * (It1 - It2)

    return Ix, Iy, It

def horn(I1,I2,alpha, N):

    A = np.array([
        [1/12,1/6,1/12],
        [1/6,0,1/6],
        [1/12,1/6,1/12]
    ])

    u = np.zeros_like(I1)
    v = np.zeros_like(I1)

    Ix, Iy, It = gradHorn(I1,I2)

    bigBot = alpha + np.power(Ix,2) + np.power(Iy,2)

    for i in range(N):

        uinv =  signal.convolve2d(u, A, boundary='symm', mode='same')
        vinv =  signal.convolve2d(v, A, boundary='symm', mode='same')

        bigTop = Ix*uinv+Iy*vinv+It
        

        big = bigTop/bigBot

        u = uinv - Ix*big
        v = vinv - Iy*big

    w = np.dstack((u,v))
    img = python.middlebury.computeColor(w)

    return w

def optimize_horn(I1, I2, wgt, min_alpha=1, max_alpha=1501, step=100, N=150):
    '''returns the parameters for which the method give the best result,
    wrt. ground truth.'''

    best_epe = float('inf')
    epe = []
    epe_std = []
    best_ang = float('inf')
    ang_std = []
    ang = []
    best_alpha=[min_alpha for i in range(3)]
    alphas = [i for i in range(1, max_alpha-min_alpha, step)] 
    
    for alpha in alphas:
        wobs = horn(I1, I2, alpha, N)
        # register errors
        e1 = endpoint_error(wobs, wgt)
        e2 = angular_error(wobs, wgt)
        epe.append( e1[0] )
        ang.append( e2[0] )
        epe_std.append( e1[1] )
        ang_std.append( e2[1] )
        # select the best alphas
        if epe[-1] < best_epe:
            best_epe = epe[-1]
            best_alpha[0] = alpha
        if ang[-1] < best_ang:
            best_ang = ang[-1]
            best_alpha[1] = alpha
       
    return alphas, best_alpha, epe, best_epe, ang, best_ang, epe_std, ang_std