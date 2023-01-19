import matplotlib.pyplot as plt
import os
from PIL import Image
from python.middlebury import readflo
import numpy as np

def get_data(index):
    root = './data/'
    filenames_with_gt = ['mysine','square','yosemite']
    wgt_filenames = ['correct_mysine.flo', 'correct_square.flo', 'correct_yos.flo']
    filenames = ['mysine','square','yos'] 
    wgt_path = os.path.join(root, filenames_with_gt[index], wgt_filenames[index])
    I1_path = os.path.join(root, filenames_with_gt[index], filenames[index]+"9.png")
    I2_path = os.path.join(root, filenames_with_gt[index], filenames[index]+"10.png")
    wgt = readflo(wgt_path)
    I1 = Image.open(I1_path).convert('L')
    I2 = Image.open(I2_path).convert('L')
    return wgt, I1, I2

def plot_errors(x, errors):
    # Draw Plot 
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']

    min_y = np.min(np.array(errors).flatten())
    max_y = np.max(np.array(errors).flatten())
    fig, ax = plt.subplots(1, 3, figsize=(16,9), dpi= 80)
    ax[0].plot(x, errors[0], label='Endpoint E', alpha=0.5, color=mycolors[1], linewidth=5)
    ax[1].plot(x, errors[1], label='Angular E', alpha=0.5, color=mycolors[2], linewidth=5)
    ax[2].plot(x, errors[2], label='Norm E', alpha=0.5, color=mycolors[0], linewidth=5)

    # Decorations
    ax[0].set_title('Endpoint Error given alpha values', fontsize=18)
    ax[1].set_title('Angular Error given alpha values', fontsize=18)
    ax[2].set_title('Norm Error given alpha values', fontsize=18)
#    ax.set(ylim=[0, 30])
#    ax.legend(loc='best', fontsize=12)
    plt.xticks(x[::200], fontsize=10, horizontalalignment='center')
#    plt.yticks(np.arange(2.5, 30.0, 2.5), fontsize=10)
    plt.xlim(-10, x[-1])

    # Draw Tick lines  
    step = (max_y - min_y) /10 
    for y in np.arange(min_y, max_y, step) :    
        plt.hlines(y, xmin=0, xmax=len(x), colors='black', alpha=0.3, linestyles="--", lw=0.5)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.show()


def endpoint_error(wobs, wgt):
    '''compute the end point error mean and std between 
    two optical_flow vectors - i.e L1'''
    epe = np.abs(wobs-wgt)
    return epe.mean(), epe.std()

def angular_error(wobs, wgt):
    '''e.g with angular error'''
    uobs = wobs[:,:,0]
    vobs = wobs[:,:,1]
    ugt = wgt[:,:,0]
    vgt = wgt[:,:,1]

    error = 1 + uobs*ugt + vobs*vgt
    divisor = np.sqrt( 1+np.power(ugt,2) + np.power(vgt,2))*(np.sqrt( 1 + np.power(uobs,2) + np.power(vobs,2)))
    ang_err = np.rad2deg(np.arccos(np.round(error/divisor, 6)))
    return np.mean(ang_err), np.std(ang_err)

def norm_error(wobs, wgt):
    '''e.g with norm error'''
    norm_wobs = np.linalg.norm(wobs.flatten(),ord=1)
    norm_wgt = np.linalg.norm(wgt.flatten(),ord=1)
    return np.abs(norm_wgt - norm_wobs).mean()

