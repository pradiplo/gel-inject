
from cProfile import label
import csv
#from turtle import title
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
import numpy as np
import re
import cv2
from os import listdir
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import shutil
from itertools import groupby
from scipy import ndimage
import math

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def binarize(img):
    level = np.max(img)/2
    bw_img = np.where(img>level, 1, 0)
    bw_img = 1 - bw_img
    return bw_img.astype("uint8")

def binarize_2(img):
    #bw_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20)
    thresh, bw_img = cv2.threshold(img, 77, 255, cv2.THRESH_BINARY)
    #bw_img = 1-bw_img
    #print(bw_img)
    #print(thresh)
    return bw_img.astype("uint8")

def find_threshold(img):
    thresh, bw_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh


def get_image(path,ori_shape):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #print(img.shape)
    img = cv2.resize(img,ori_shape[::-1],interpolation = cv2.INTER_AREA)
    img_dn = cv2.fastNlMeansDenoising(img)
    #img_dn = img
    #img_bin = binarize(img_dn)
    #img_bin = img_bin[bound[0]:bound[1],bound[2]:bound[3]]
    return img_dn

def clean_island(img):
    #plt.imshow(img,cmap="gray")
    #plt.show()
    labeled, Nlabels = ndimage.label(img)
    label_size = [(labeled == label).sum() for label in range(Nlabels + 1)]
    
    for label,size in enumerate(label_size):
        #print(size)
        if size < 7:
            #print(label)
            #print(size)
            #print(img[labeled])
            img[labeled == label] = 0
    return img        

def get_cav(path,ori_shape,bound,step):
    img_dn = get_image(path, ori_shape)
    img_print = img_dn[bound[0]:bound[1],bound[2]:bound[3]]
    #plt.imshow(img_print,cmap="gray")
    #plt.savefig()
    if step >= 0:
        img_bin = cv2.bitwise_not(binarize_2(img_dn))
        img_bin = img_bin[bound[0]:bound[1],bound[2]:bound[3]]

        img_bin = clean_island(img_bin)
        #
    else:
        img_tmp = img_dn[bound[0]:bound[1],bound[2]:bound[3]]
        shape = img_tmp.shape
        #print(shape)
        img_bin= np.full(shape,0)
        #print(img_bin)
        #  

    #plt.imshow(img_bin,cmap="gray")
    #plt.show()

    black_idx = np.where(img_bin==255)
    #if len(black_idx[0]) > 0:
    #   print(len(black_idx[0]))
    #   print(max(black_idx[0]))
    #print(black_idx$)
    return black_idx,img_print

def get_bound_from_edges(whites_idx):
    idx_z = whites_idx[0]
    idx_x = whites_idx[1]
    df_index = pd.DataFrame(np.column_stack([idx_z,idx_x]), columns=["idx_z","idx_x"]) 
    bin = pd.cut(df_index["idx_z"],50) #group every 10 pixels in z, find group with most white pixels
    a = df_index.groupby(bin).size().nlargest(2).index.values 
    top_idx =int(min(a[0].right, a[1].right))
    return top_idx + 155

def get_top_bound(path,dnpath):
    deltan = np.loadtxt(dnpath,delimiter=",")
    ori_shape =deltan.shape
    img_dn = get_image(path, ori_shape)
    start_idx = 0
    img_dn = img_dn[start_idx:-1,0:96]
    #plt.imshow(img_dn,cmap="gray")
    edges = cv2.Canny(img_dn,50,50)
    whites_idx= np.where(edges>1)
    #plt.imshow(edges,cmap="gray")
    #plt.show()
    top_idx = get_bound_from_edges(whites_idx)
    #print(top_idx)
    return top_idx

def csv_to_png(path,bwpath,savepath,top_idx):   
    deltan = np.loadtxt(path,delimiter=",")
    ori_size = deltan.shape
    #top_idx = get_top_bound(bwpath,ori_size)
    #print(top_idx)
    bound = [top_idx, top_idx+200, 0, 96]
    deltan = deltan[bound[0]:bound[1],bound[2]:bound[3]]
    scale = 2*1.3100436681222708e-02
    mask_idx,img_print = get_cav(bwpath, ori_size,bound)
    deltan[mask_idx] = 0
    #deltan = cv2.fastNlMeansDenoisingColored(deltan)
    plt.rcParams['figure.figsize'] = (8, 12)
    plt.figure()
    ax = plt.subplot()
    im = ax.imshow(deltan,vmin=0,vmax=120,cmap="viridis",extent=[-48*scale,48*scale,-200*scale,0],aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r"$\Delta n$ (nm)", rotation=270, fontsize=32,labelpad=30)
    ax.set_xlabel(r"$x$ (mm)",fontsize=32)
    ax.set_ylabel(r"$z$ (mm)",fontsize=32)
    ax.set_xticks([-1.2, -0.6, 0, 0.6,1.2])
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.tick_params(axis='both', which='minor', labelsize=32)
    cbar.ax.tick_params(axis='both', which='major', labelsize=32)
    cbar.ax.tick_params(axis='both', which='minor', labelsize=32)
    plt.tight_layout()
    #plt.show()
    plt.savefig(savepath,bbox_inches='tight')
    return 0

def deln_to_png(deltan,savepath):   
    scale = 2*1.3100436681222708e-02
    #deltan = cv2.fastNlMeansDenoisingColored(deltan)
    plt.rcParams['figure.figsize'] = (8, 12)
    plt.figure()
    ax = plt.subplot()
    im = ax.imshow(deltan,vmin=0,vmax=120,cmap="viridis",extent=[-48*scale,48*scale,-200*scale,0],aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r"$\Delta n$ (nm)", rotation=270, fontsize=32,labelpad=30)
    ax.set_xlabel(r"$x$ (mm)",fontsize=32)
    ax.set_ylabel(r"$z$ (mm)",fontsize=32)
    ax.set_xticks([-1.2, -0.6, 0, 0.6,1.2])
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.tick_params(axis='both', which='minor', labelsize=32)
    cbar.ax.tick_params(axis='both', which='major', labelsize=32)
    cbar.ax.tick_params(axis='both', which='minor', labelsize=32)
    plt.tight_layout()
    #plt.show()
    plt.savefig(savepath,bbox_inches='tight')
    return 0

def phi_to_png(phi,savepath):   
    scale = 2*1.3100436681222708e-02
    #deltan = cv2.fastNlMeansDenoisingColored(deltan)
    plt.rcParams['figure.figsize'] = (8, 12)
    plt.figure()
    ax = plt.subplot()
    im = ax.imshow(phi,cmap="viridis",vmin=0,vmax=180,extent=[-48*scale,48*scale,-200*scale,0],aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r"$\phi$ (deg)", rotation=270, fontsize=32,labelpad=30)
    ax.set_xlabel(r"$x$ (mm)",fontsize=32)
    ax.set_ylabel(r"$z$ (mm)",fontsize=32)
    ax.set_xticks([-1.2, -0.6, 0, 0.6,1.2])
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.tick_params(axis='both', which='minor', labelsize=32)
    cbar.ax.tick_params(axis='both', which='major', labelsize=32)
    cbar.ax.tick_params(axis='both', which='minor', labelsize=32)
    plt.tight_layout()
    #plt.show()
    plt.savefig(savepath,bbox_inches='tight')
    return 0

def stress_to_png(sig,savepath):   
    scale = 2*1.3100436681222708e-02
    #deltan = cv2.fastNlMeansDenoisingColored(deltan)
    plt.rcParams['figure.figsize'] = (8, 12)
    plt.figure()
    ax = plt.subplot()
    im = ax.imshow(sig,cmap="viridis",vmin=-8e12,vmax=8e12)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r"$\sigma_{rz}$", rotation=270, fontsize=32,labelpad=30)
    #ax.set_xlabel(r"$x$ (mm)",fontsize=32)
    #ax.set_ylabel(r"$z$ (mm)",fontsize=32)
    #ax.set_xticks([-1.2, -0.6, 0, 0.6,1.2])
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.tick_params(axis='both', which='minor', labelsize=32)
    cbar.ax.tick_params(axis='both', which='major', labelsize=32)
    cbar.ax.tick_params(axis='both', which='minor', labelsize=32)
    plt.tight_layout()
    #plt.show()
    plt.savefig(savepath,bbox_inches='tight')
    return 0


def to_png(img,savepath):   
    #deltan = cv2.fastNlMeansDenoisingColored(deltan)
    scale = 1.3100436681222708e-02 * 2
    plt.rcParams['figure.figsize'] = (8, 12)
    plt.figure()
    ax = plt.subplot()
    im = ax.imshow(img,cmap="gray",extent=[-48*scale,48*scale,-200*scale,0],aspect="auto")
    divider = make_axes_locatable(ax)
    ax.set_xlabel(r"$x$ (mm)",fontsize=32)
    ax.set_ylabel(r"$z$ (mm)",fontsize=32)
    ax.set_xticks([-1.2, -0.6, 0, 0.6,1.2])
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.tick_params(axis='both', which='minor', labelsize=32)
    plt.tight_layout()
    #plt.show()
    plt.savefig(savepath,bbox_inches='tight')
    return 0


def get_depth_width(whites):
    if len(whites[0]) > 0:
        depth = max(whites[0])
        width = max(whites[1]) - min(whites[1])
    else:
        depth = 0
        width = 0
    return depth, width

def integrate_vol(whites,scale):
    if len(whites[0]) > 0:
        idx_z = whites[0].flatten()
        idx_x = whites[1].flatten()
        df_index = pd.DataFrame(np.column_stack([idx_z,idx_x]), columns=["idx_z","idx_x"])
        a = df_index.groupby("idx_z").size().to_numpy() 
        dv = np.pi * np.power(a,2) * np.power(scale,3)
        vol = np.sum(dv)
    else:
        vol = 0
    return vol

def get_data(path,bwpath,axis_path,top_idx,i, half):
    deltan = np.loadtxt(path,delimiter=",")
    phi = np.loadtxt(axis_path,delimiter=",")
    ori_size = deltan.shape
    if half == True:
        bound = [top_idx, top_idx+200, 48, 96] #right
    else:
        bound = [top_idx, top_idx+200, 0, 96]   

    deltan = deltan[bound[0]:bound[1],bound[2]:bound[3]]
    phi = phi[bound[0]:bound[1], bound[2]:bound[3]]

    mask_idx,imgprint = get_cav(bwpath, ori_size,bound,i)

    deltan[mask_idx] =np.nan
    phi[mask_idx] = np.nan

    return deltan, phi

def tomography_op_r(deltan,phi):
    phi =  np.radians(phi)
    deltan = deltan*1e-9 #nm to m
    coef = 1e-9
    scale = 2*1.3100436681222708e-05
    xn = int(deltan.shape[1]/2) -1
    zn = deltan.shape[0]
    yn = xn

    x_axis = np.linspace(1,xn+1,xn+1)*scale
    z_axis = np.linspace(1,zn+1,zn+1)*scale*(-1)
    y_axis = x_axis
   
    dx = x_axis[1] - x_axis[0]
    dz = z_axis[1] - z_axis[0]

    v1 = deltan[0:200, 48:96] * np.cos(2*phi[0:200, 48:96])
    v2 = deltan[0: 200, 48:96] * np.sin(2*phi[0:200, 48:96])

    
    ly = np.zeros((yn, xn))
    r_re = np.zeros((yn, xn))
    theta_re = np.zeros((yn, xn))

    for i in range(xn):
        for k in range(yn):
            r_re[k, i] = round(np.sqrt(x_axis[i] ** 2 + y_axis[k] ** 2) / scale)
            ly[k, i] = scale * np.count_nonzero(r_re[:, i] == k +1)
            theta_re[k, i] = np.arctan(x_axis[i] / y_axis[k])

    o_rz_re = np.zeros((zn, xn))
    for j in range(zn):
        #print(j)
        for i in range(xn):
            #print(i)
            sum_o_rz = 0
            for k in range(i):
                #print(k)
                if k == 0:
                    pass
                else:
                    sum_o_rz += ly[i,k ] * o_rz_re[j,i] * np.cos(theta_re[i,k ])
            o_rz_re[j, i] = (v2[j, i] - 4 * coef * sum_o_rz) / (4 * coef * ly[i,i])
    """
    #o_rz_re = o_rz_re[:,:xn]
    dV2 = v2[:-1, :] - v2[1:, :]
    dV2 = np.vstack((dV2, np.zeros((1, xn))))

# Initialize o_zz_re
    o_zz_re = np.zeros((zn, xn+1))

# Loop through j and i
    for j in range(zn):
        for i in range(1,xn+1):
            sum_V2 = dV2[j, xn - i]
            sum_o_zz = 0
            for k in range(1, i):
                if k > 1:
                    sum_V2 += dV2[j, xn - k]
                    sum_o_zz += ly[xn - k, xn - i] * o_zz_re[j, xn - k]

            o_zz_re[j, xn - i] = (sum_V2 * dx / (2 * dz) - v1[j, xn - i] - 2 * coef * sum_o_zz) / (2 * coef * ly[xn - i, xn - i])
    """
    return -o_rz_re/1000, ly, r_re, theta_re, 0, o_rz_re,v2



def tomography_op_l(deltan,phi):
    phi[0:200,0:48] = phi[0:200,0:48] * (-1)
    phi =  np.radians(phi)
    deltan = deltan*1e-9 #nm to m
    coef = 1e-9
    scale = 2*1.3100436681222708e-05
    xn = int(deltan.shape[1]/2) -1
    zn = deltan.shape[0]
    yn = xn

    x_axis = np.linspace(1,xn+1,xn+1)*scale
    z_axis = np.linspace(1,zn+1,zn+1)*scale*(-1)
    y_axis = x_axis
   
    dx = x_axis[1] - x_axis[0]
    dz = z_axis[1] - z_axis[0]

    v1 = np.fliplr(deltan[0:200, 0:48] * np.cos(2*phi[0:200, 0:48]))
    v2 = np.fliplr(deltan[0:200, 0:48] * np.sin(2*phi[0:200, 0:48]))

    #print(v2.shape)
    ly = np.zeros((yn, xn))
    r_re = np.zeros((yn, xn))
    theta_re = np.zeros((yn, xn))

    for i in range(xn):
        for k in range(yn):
            r_re[k, i] = round(np.sqrt(x_axis[i] ** 2 + y_axis[k] ** 2) / scale)
            ly[k, i] = scale * np.count_nonzero(r_re[:, i] == k +1)
            theta_re[k, i] = np.arctan(x_axis[i] / y_axis[k])

    o_rz_re = np.zeros((zn, xn))
    for j in range(zn):
        #print(j)
        for i in range(xn):
            #print(i)
            sum_o_rz = 0
            for k in range(i):
                #print(k)
                if k == 0:
                    pass
                else:
                    sum_o_rz += ly[i,k ] * o_rz_re[j,i] * np.cos(theta_re[i,k ])
            o_rz_re[j, i] = (v2[j, i] - 4 * coef * sum_o_rz) / (4 * coef * ly[i,i])
    """
    #o_rz_re = o_rz_re[:,:xn]
    dV2 = v2[:-1, :] - v2[1:, :]
    dV2 = np.vstack((dV2, np.zeros((1, xn))))

# Initialize o_zz_re
    o_zz_re = np.zeros((zn, xn+1))

# Loop through j and i
    for j in range(zn):
        for i in range(1,xn+1):
            sum_V2 = dV2[j, xn - i]
            sum_o_zz = 0
            for k in range(1, i):
                if k > 1:
                    sum_V2 += dV2[j, xn - k]
                    sum_o_zz += ly[xn - k, xn - i] * o_zz_re[j, xn - k]

            o_zz_re[j, xn - i] = (sum_V2 * dx / (2 * dz) - v1[j, xn - i] - 2 * coef * sum_o_zz) / (2 * coef * ly[xn - i, xn - i])
    """
    return -o_rz_re/1000, ly, r_re, theta_re, 0, o_rz_re,v2


def tomography_2(deltan,phi):
    phi =  np.radians(phi) 
    deltan = deltan
    coef = 1e-9
    scale = 1.3100436681222708e-05 * 2

    xn = int(deltan.shape[1] /2)
    zn = deltan.shape[0]
    yn = xn

    x_axis = np.linspace(0,xn,xn)*scale 
    z_axis = np.linspace(0,zn,zn)*scale
    y_axis = x_axis
   
    dx = x_axis[1] - x_axis[0]
    dz = z_axis[1] - z_axis[0]

    v1 = deltan[0:200, 48:96] * np.cos(2*phi[0:200, 48:96])
    v2 = deltan[0:200, 48:96] * np.sin(2*phi[0:200, 48:96])

    
    x_n, y_n = np.meshgrid(np.arange(1,xn+1), np.arange(1,yn))
    r_re = np.sqrt(x_n**2 + y_n**2)

    plt.imshow(r_re)
    plt.show()



    A_rz = x_n / r_re  
    A_zz = r_re / r_re  

    alpha_rz = np.zeros((yn, xn))
    alpha_zz = np.zeros((yn, xn))

    #x_n, y_n = np.meshgrid(np.arange(1, xn + 1), np.arange(1, yn + 1))
    x_n, y_n = np.meshgrid(np.arange(xn), np.arange(xn))
    r_re = np.sqrt(x_n**2 + y_n**2)

    A_rz = x_n / r_re  # Equivalent to cos(theta) in MATLAB (alpha_rz)
    A_zz = np.ones_like(r_re)  # Equivalent to 1 in MATLAB (alpha_zz)

    alpha_rz = np.zeros((yn, xn))
    alpha_zz = np.zeros((yn, xn))

    for j in range(xn):
        min_r = int(np.floor(np.min(r_re[:, j])))
        for radius in range(min_r, int(np.floor(np.max(r_re[:, j])))):

            if radius >= xn:
                break

            judge_min = (1 + np.floor(r_re[:, j]) - r_re[j, :]) * (np.floor(r_re[:, j]) == radius)
            judge_max = (r_re[j, :] - np.floor(r_re[:, j])) * (np.floor(r_re[:, j]) == radius)

            alpha_rz[j, radius] = np.sum(A_rz[:, j] * judge_min)
            alpha_zz[j, radius] = np.sum(A_zz[:, j] * judge_min) + np.sum(A_zz[:, j] * judge_max)

    alpha_rz = 4 * coef * alpha_rz.T * scale + 2 * coef * np.eye(xn) * scale        
    

    sig_rz = np.zeros((zn, xn))
    #from scipy import linalg

    #a = np.linalg.cond(alpha_rz)
    #print(a)
    #print(alpha_rz.shape)
    #print(np.amax(alpha_rz))
    #print(np.amin(alpha_rz))
    #for j in range(zn):
    #    sig_rz[j,:] = np.linalg.solve(alpha_rz.T, v2[j, :])
    #print(v2)    


    return sig_rz,alpha_rz, v2 #sig_rz                   

def tomography(deltan,phi):
    phi =  np.radians(phi) - np.pi /2
    deltan = deltan*1e-9
    coef = 1e-9
    scale = 1.3100436681222708e-02 * 2
    xn = int(deltan.shape[1] /2)
    zn =deltan.shape[0]
    yn = xn

    x_axis = np.linspace(0,xn,xn)*scale 
    z_axis = np.linspace(0,zn,zn)*scale
    y_axis = x_axis
   
    dx = x_axis[1] - x_axis[0]
    dz = z_axis[1] - z_axis[0]

    v1 = deltan[0:200, 48:96] * np.cos(2*phi[0:200, 48:96])
    v2 = deltan[0:200, 48:96] * np.sin(2*phi[0:200, 48:96])

    x_n, y_n = np.meshgrid(np.arange(1,xn+1), np.arange(1,yn +1))
    r_re = np.sqrt(x_n**2 + y_n**2)
    print(r_re)
    A_rz = x_n / r_re  
    A_zz = r_re / r_re  

    alpha_rz = np.zeros((yn, xn))
    alpha_zz = np.zeros((yn, xn))

    for j in range(xn):
        min_r = int(np.floor(np.min(r_re[:, j])))
        #print(int(np.floor(np.max(r_re[:, j]))))
        #if int(np.floor(np.max(r_re[:, j]))) < xn:
            
        for radius in range(min_r, int(np.floor(np.max(r_re[:, j])))):
            if radius < xn:
                judge_min = np.zeros((xn,1))
                judge_max = np.zeros((xn,1))
                #print(radius)

                for i in range(xn):
                    #print(np.floor(r_re[i, j]))
                    if np.floor(r_re[i, j]) == radius:
                        judge_min[i] = 1 + np.floor(r_re[i, j]) - r_re[j, i]
                        judge_max[i] = r_re[j, i] - np.floor(r_re[i, j])    
                #print(judge_min)
                alpha_rz_1 = np.sum(A_rz[:, j] * judge_min)
                alpha_rz_2 = np.sum(A_rz[:, j] * judge_max)
                alpha_rz[j, radius] = alpha_rz_1 + alpha_rz_2

                alpha_zz_1 = np.sum(A_zz[:, j] * judge_min)
                alpha_zz_2 = np.sum(A_zz[:, j] * judge_max)
                alpha_zz[j, radius] = alpha_zz_1+ alpha_zz_2

    alpha_rz = 4 * coef * alpha_rz.T * scale + 2 * coef * np.eye(xn) * scale
    alpha_zz = 2 * coef * alpha_zz.T * scale + 1 * coef * np.eye(xn) * scale

    sig_rz = np.zeros((zn, xn))
    #v1 = np.nan_to_num(v1)
    #a = np.linalg.cond(v1)
    #print(a)
    for j in range(zn):

        sig_rz[j, :] = np.linalg.solve(alpha_rz, v2[j, :])

    return sig_rz, alpha_rz.T, v2

def reconstruction_loop(main_dir, work_dirs):
    for work_dir in work_dirs:

        cases = [i for i in listdir( main_dir + work_dir  ) if i.startswith("2")]

        for case in cases:

            bw_dir = main_dir + work_dir + case
            save_dir = main_dir + work_dir + case + "/movie/"

            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)
            
            bwfiles = [i for i in sorted_nicely(listdir(bw_dir )) if i.endswith(".tif")]
            csv_dir = [i for i in listdir(main_dir + work_dir + case + "/CSVs/") if not i.startswith(".")]
            deln_dir = main_dir + work_dir + case  + "/CSVs/" +csv_dir[0]
            deln_files = [i for i in sorted_nicely(listdir(deln_dir)) if i.endswith(".csv") and "axis" not in i ]
            phi_files = [i for i in sorted_nicely(listdir(deln_dir)) if i.endswith(".csv") and "retardation" not in i]
            print(main_dir + work_dir + case)

            stress = []
            for i in range(60):
                dn_path = deln_dir + "/" + deln_files[i]
                phi_path =  deln_dir + "/" + phi_files[i]
                bwpath = bw_dir + "/" + bwfiles[i]
                #print(spath)
                top_idx = get_top_bound(bw_dir + "/" + bwfiles[0],deln_dir  + "/"+ deln_files[0])
                deltan, phi = get_data(dn_path,bwpath,phi_path,top_idx,i,False)
                #deln_to_png(deltan, bw_dir + "/deltan_" + str(i).zfill(4) + ".png")
                #phi_to_png(phi, bw_dir + "/phi_" + str(i).zfill(4) + ".png")
                #sigrz,alp,v2 = tomography_2(deltan,phi)
                sigrz_r,ly,r, theta,sigzz,ratio, v2_r = tomography_op_r(deltan,phi)
                sigrz_l,ly,r, theta,sigzz,ratio, v2_l = tomography_op_l(deltan,phi)
                #print(sigrz)
                sigrz = np.hstack((np.fliplr(sigrz_l),sigrz_r))
                v2 = np.hstack((np.fliplr(v2_l),v2_r))
                #sigrz = np.where(np.abs(sigrz) > 1e30, np.nan, sigrz)
                #print(alp)
                #print(np.nanmin(sigrz))
                #print(np.nanmax(sigrz))

                im = plt.imshow(sigrz,vmin=-2e2, vmax=2e2,cmap="viridis")
                plt.colorbar(im)
                #plt.imshow(v2)
                #plt.imshow(sigrz)
                plt.show()
                #plt.show()
                #plt.savefig("./data/stress_"  + str(i).zfill(4) +".png")
                #print(sigrz)
                #print(sigrz.shape)
                #stress_to_png(sigrz, bw_dir + "/stress_" + str(i).zfill(4) + ".png")
                #stress.apend(sigrz)
def get_stats(path,bwpath,top_idx,i):   
    deltan = np.loadtxt(path,delimiter=",")
    ori_size = deltan.shape
    #print(ori_size)
    bound = [top_idx, top_idx+200, 0, 96]
    deltan = deltan[bound[0]:bound[1],bound[2]:bound[3]]
    scale = 1.3100436681222708e-02 * 2
    mask_idx,img_print = get_cav(bwpath, ori_size,bound,i)
    #whites = get_whites(bwpath, ori_size,bound)
    depth, width = get_depth_width(mask_idx)
    true_vol = integrate_vol(mask_idx,scale)
    deltan[mask_idx] = 0
    #plt.imshow(deltan,vmin=0,vmax=120,cmap="viridis")
    #plt.show()
    deln_avg = np.mean(deltan)
    vol_cyl = 0.25*np.pi * np.power((width*scale),2)* (depth*scale)
    return depth*scale, width*scale, deln_avg, vol_cyl,true_vol, img_print

def get_spatial_avg(path,bwpath,ymin,ymax):
    files = [i for i in sorted_nicely(listdir(path)) if i.endswith(".csv")]
    bwfiles = [i for i in sorted_nicely(listdir(bwpath)) if i.endswith(".tif")]
    #print(bwfiles)
    deln = []
    deltat = 1/25000
    for f,b in zip(files,bwfiles):
        deltan = np.loadtxt(path + f,delimiter=",")
        ori_shape = deltan.shape
        deltan = deltan[ymin:ymax,0:96]
        bound = [ymin,ymax,0,96]
        #print(bwpath + b)
        idx_mask = get_cav(bwpath + b,ori_shape,bound)   
        avg = np.mean(deltan)
        deln.append(avg)
    time = np.linspace(0,len(deln),len(deln))*deltat*1000   
    return time, deln

def main_calculation(main_dir, work_dirs):
    # tuple to store mean and std
    volumes = ([],[]) 
    cavityDepth = ([],[])
    cavityMaxWidth =([],[])
    avgDeltan = [[],[]]

    for work_dir in work_dirs:

        cases = [i for i in listdir( main_dir + work_dir  ) if i.startswith("2")]

        depthss =[]  #np.empty(60,dtype=np.float64)
        widthss =[] #np.empty(60,dtype=np.float64)
        delnss =[] #np.empty(60,dtype=np.float64)
        volss = [] #np.empty(60,dtype=np.float64)

        for case in cases:

            bw_dir = main_dir + work_dir + case
            save_dir = main_dir + work_dir + case + "/movie/"

            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)
            
            bwfiles = [i for i in sorted_nicely(listdir(bw_dir )) if i.endswith(".tif")]
            csv_dir = [i for i in listdir(main_dir + work_dir + case + "/CSVs/") if not i.startswith(".")]
            deln_dir = main_dir + work_dir + case  + "/CSVs/" +csv_dir[0]
            files = [i for i in sorted_nicely(listdir(deln_dir)) if i.endswith(".csv") and "axis" not in i ]
        
            depths = []
            widths = []
            delns  = []
            vols = []   

            print(main_dir + work_dir + case)

            for i in range(1,60):
                path = deln_dir + "/" + files[i]
                bwpath = bw_dir + "/" + bwfiles[i]
                #print(spath)
                top_idx = get_top_bound(bw_dir + "/" + bwfiles[0],deln_dir  + "/"+ files[0])
                depth, width, deln_avg,vol_cyl,vol, img_print = get_stats(path,bwpath,top_idx,i)
                #spath = save_dir + "deltan" + "_" + str(i).zfill(4) + ".png"
                #csv_to_png(path,bwpath,spath,top_idx)
                #plt.imshow(img_print,cmap="gray")
                to_png(img_print, bw_dir + "/cavity_" + str(i).zfill(4) + ".png" )
                depths.append(depth)
                widths.append(width)
                delns.append(deln_avg)
                vols.append(vol)

            depths = np.array(depths)
            widths = np.array(widths)
            delns = np.array(delns)
            vols = np.array(vols)
            plot_single(vols,bw_dir)

            depthss.append(depths) # = np.column_stack((depthss,depths))
            widthss.append(widths) # = np.column_stack((widthss,widths))
            delnss.append(delns) # = np.column_stack((delnss,delns))
            volss.append(vols) # = np.column_stack((volss,vols))

        
        depthss =np.array(depthss).T
        #print(depthss.shape)
        widthss = np.array(widthss).T
        delnss = np.array(delnss).T
        volss = np.array(volss).T

        volumes[0].append(np.mean(volss,axis=1))
        volumes[1].append(np.std(volss, axis=1))  
        cavityDepth[0].append(np.mean(depthss,axis=1))
        cavityDepth[1].append(np.std(depthss, axis=1))
        cavityMaxWidth[0].append(np.mean(widthss,axis=1))
        cavityMaxWidth[1].append(np.std(widthss, axis=1))
        avgDeltan[0].append(np.mean(delnss,axis=1))
        avgDeltan[1].append(np.std(delnss, axis=1)) 
    
    return volumes, cavityDepth, cavityMaxWidth , avgDeltan      

#def stress_reconstruct(Delta_exp_left,Phi_exp_left):
#   delta_exp_left 



def plot_single(y,savepath):
    deltat = 1/25000
    time = np.linspace(0,len(y),len(y))*deltat*1000 # in ms
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.figure()
    plt.plot(time,y)
    plt.xlabel(r'$t$ (ms)',fontsize=40)
    plt.ylabel("Quantity",fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=32)
    plt.tick_params(axis='both', which='minor', labelsize=32)
    #plt.legend(loc="best",prop={'size': 20}, framealpha=0.5)
    plt.tight_layout()
    plt.savefig(savepath + "/test.png")

def create_plots(tup, ylab,filename,limit):
    v_means, v_std = tup
    deltat = 1/25000
    time = np.linspace(0,len(v_means[0]),len(v_means[0]))*deltat*1000 # in ms
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.figure()
    plt.errorbar(time[::2], v_means[0][::2],yerr=v_std[0][::2],fmt="-s",linewidth=2,elinewidth=1, label="3w.t%")
    plt.errorbar(time[::2], v_means[1][::2],yerr=v_std[1][::2],fmt="-s",linewidth=2,elinewidth=1, label="5w.t%")
    plt.errorbar(time[::2], v_means[2][::2],yerr=v_std[2][::2],fmt="-s",linewidth=2,elinewidth=1, label="7w.t%")
    plt.xlabel(r'$t$ (ms)',fontsize=40)
    plt.ylabel(ylab,fontsize=40)
    plt.xlim(0,2.)
    plt.ylim(limit[0],limit[1])
    plt.tick_params(axis='both', which='major', labelsize=32)
    plt.tick_params(axis='both', which='minor', labelsize=32)
    plt.legend(loc="best",prop={'size': 20}, framealpha=0.5)
    plt.tight_layout()
    plt.savefig("./data/" +  filename   +  ".png")



if __name__ == '__main__':
    main_dir = "/Users/tagawayo-lab-pc43/workspace/tagawalab/amed/gel-inject/data/"
    #work_dirs =  ["3%_crysta/","5%_crysta/","7%_crysta/"]
    work_dirs =  ["5%_crysta/"]
    reconstruction_loop(main_dir, work_dirs)
    # cavity dinamics +  delta_n
    #volumes, cavityDepth, cavityMaxWidth , avgDeltan  = main_calculation(main_dir, work_dirs)
    
    #create_plots(volumes, r'$V (\rm mm^3)$ ', "volumes_3", (0,25))
    #create_plots(cavityDepth, r'$ D (\rm mm)$ ', "depth_3", (0,6))
    #create_plots(cavityMaxWidth, r'$W_{\rm max} (\rm mm)$ ', "width_3", (0,3))
    #create_plots(avgDeltan, r'$\Delta n_{\rm avg} (\rm nm)$ ', "deltan", (0,50))
