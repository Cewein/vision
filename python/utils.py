import matplotlib.pyplot as plt
import os
from PIL import Image
from python.middlebury import readflo, computeColor
import numpy as np
from python.horn import optimize_horn, horn
from python.lucasKanade import optimize_LK, lucasKanade
from tqdm import tqdm

def get_data(index):
    root = './data/'
    filenames_with_gt = ['mysine','square','rubberwhale', 'yosemite']
    wgt_filenames = ['correct_mysine.flo', 'correct_square.flo', 'correct_rubberwhale10.flo', 'correct_yos.flo']
    filenames = ['mysine','square','frame','yos'] 
    wgt_path = os.path.join(root, filenames_with_gt[index], wgt_filenames[index])

    I1_path = os.path.join(root, filenames_with_gt[index], filenames[index]+"9.png")
    I2_path = os.path.join(root, filenames_with_gt[index], filenames[index]+"10.png")

    if(filenames[index] == 'frame'):
        I1_path = os.path.join(root, filenames_with_gt[index], filenames[index]+"10.png")
        I2_path = os.path.join(root, filenames_with_gt[index], filenames[index]+"11.png")
        
    wgt = readflo(wgt_path)
    I1 = Image.open(I1_path).convert('L')
    I2 = Image.open(I2_path).convert('L')
    return wgt, I1, I2



def plot_all_horn():
    # Draw Plot 
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']

    fig, ax = plt.subplots(4, 6, figsize=(30,15), dpi= 80)
    for i in tqdm(range(4)):
        # compute + errors
        wgt, I1, I2 = get_data(i)
        alphas, best_alphas, epe, best_epe, ang, best_ang, epe_std, ang_std = optimize_horn(I1, I2, wgt, max_alpha=5001)
        w_epe = horn(I1, I2, alpha=best_alphas[0], N=300)
        w_ang =  horn(I1, I2, alpha=best_alphas[1], N=300)
        # plots

        ax[i,0].plot(alphas, ang, label='std : '+str(np.array(ang_std).mean()), alpha=0.8, color=mycolors[3], linewidth=3)
        ax[i,1].plot(alphas, epe, label='std : '+str(np.array(epe_std).mean()), alpha=0.8, color=mycolors[4], linewidth=3)
        ax[i,0].legend()
        ax[i,1].legend()
        ax[i,2].imshow(computeColor(w_ang))
        ax[i,3].imshow(computeColor(w_epe))
        ax[i,4].quiver(np.arange(0, w_epe.shape[1], 5), np.arange(w_epe.shape[0], 0, -5), w_epe[::5, ::5, 0], -w_epe[::5, ::5, 1])
        ax[i,5].imshow(computeColor(wgt))
        
        
        # Decorations
        ax[0,1].set_title('Endpoint Error given alpha values')
        ax[0,0].set_title('Angular Error given alpha values')
        ax[0,2].set_title('Best Endpoint Output')
        ax[0,3].set_title('Best Angular Output')
        ax[0,4].set_title('Velocity Map')
        ax[0,5].set_title('Ground_truth')
        ax[i,2].axis('off')
        ax[i,3].axis('off')
        ax[i,4].axis('off')
#     plt.xticks(x[::200], fontsize=10, horizontalalignment='center')
#    plt.yticks(np.arange(2.5, 30.0, 2.5), fontsize=10)
#     plt.xlim(-10, x[-1])

    # Draw Tick lines  
#     step = (max_y - min_y) /10 
#     for y in np.arange(min_y, max_y, step) :    
#         plt.hlines(y, xmin=0, xmax=len(x), colors='black', alpha=0.3, linestyles="--", lw=0.5)

#     # Lighten borders
#     plt.gca().spines["top"].set_alpha(0)
#     plt.gca().spines["bottom"].set_alpha(.3)
#     plt.gca().spines["right"].set_alpha(0)
#     plt.gca().spines["left"].set_alpha(.3)
    plt.show()

def plot_all_LK(func):
    # Draw Plot 
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']

    fig, ax = plt.subplots(4, 6, figsize=(30,15), dpi= 80)
    for i in tqdm(range(4)):
        # compute + errors
        wgt, I1, I2 = get_data(i)
        alphas, best_alphas, epe, best_epe, ang, best_ang, epe_std, ang_std = optimize_LK(func,I1, I2, wgt,3,30)
        w_epe = func(I1, I2, best_alphas[0])
        w_ang =  func(I1, I2, best_alphas[1])
        # plots

        angArray = np.array(ang_std)
        epeArray = np.array(epe_std)

        angArray[~np.isfinite(angArray)] = 0.0
        epeArray[~np.isfinite(epeArray)] = 0.0

        ax[i,0].plot(alphas, ang, alpha=0.8, color=mycolors[3], linewidth=3)
        ax[i,1].plot(alphas, epe, alpha=0.8, color=mycolors[4], linewidth=3)
        ax[i,2].imshow(computeColor(w_ang))
        ax[i,3].imshow(computeColor(w_epe))
        ax[i,4].quiver(np.arange(0, w_epe.shape[1], 5), np.arange(w_epe.shape[0], 0, -5), w_epe[::5, ::5, 0], -w_epe[::5, ::5, 1])
        ax[i,5].imshow(computeColor(wgt))
        
        
        # Decorations
        ax[0,1].set_title('Endpoint Error with given windows')
        ax[0,0].set_title('Angular Error with given windows')
        ax[0,2].set_title('Best Endpoint Output')
        ax[0,3].set_title('Best Angular Output')
        ax[0,4].set_title('Velocity Map')
        ax[0,5].set_title('Ground_truth')
        ax[i,2].axis('off')
        ax[i,3].axis('off')
        ax[i,4].axis('off')
    plt.show()


def plot_results(w_epe, w_ang, w_gt):
    fig, ax = plt.subplots(1,3, figsize=(20,20))
    ax[0].imshow(computeColor(w_epe))
    ax[1].imshow(computeColor(w_ang))
    ax[0].set_title('Best Endpoint Output')
    ax[1].set_title('Best Angular Output')
    ax[2].imshow(computeColor(w_gt))
    ax[2].set_title('Ground_truth')
    plt.show()
    
    

