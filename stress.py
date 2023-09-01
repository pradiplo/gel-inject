
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
    
    if step >= 1:
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
    return black_idx

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
    scale = 1.3100436681222708e-02
    mask_idx = get_cav(bwpath, ori_size,bound)
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
    ax.set_xticks([-0.6, -0.3, 0, 0.3,0.6])
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.tick_params(axis='both', which='minor', labelsize=32)
    cbar.ax.tick_params(axis='both', which='major', labelsize=32)
    cbar.ax.tick_params(axis='both', which='minor', labelsize=32)
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

def get_stats(path,bwpath,top_idx,i):   
    deltan = np.loadtxt(path,delimiter=",")
    ori_size = deltan.shape
    #print(ori_size)
    bound = [top_idx, top_idx+200, 0, 96]
    deltan = deltan[bound[0]:bound[1],bound[2]:bound[3]]
    scale = 1.3100436681222708e-02
    mask_idx = get_cav(bwpath, ori_size,bound,i)
    #whites = get_whites(bwpath, ori_size,bound)
    depth, width = get_depth_width(mask_idx)
    true_vol = integrate_vol(mask_idx,scale)
    deltan[mask_idx] = 0
    #plt.imshow(deltan,vmin=0,vmax=120,cmap="viridis")
    #plt.show()
    deln_avg = np.mean(deltan)
    vol_cyl = 0.25*np.pi * np.power((width*scale),2)* (depth*scale)
    return depth*scale, width*scale, deln_avg, vol_cyl,true_vol

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

            for i in range(0,60):
                path = deln_dir + "/" + files[i]
                bwpath = bw_dir + "/" + bwfiles[i]
                #print(spath)
                top_idx = get_top_bound(bw_dir + "/" + bwfiles[0],deln_dir  + "/"+ files[0])
                depth, width, deln_avg,vol_cyl,vol = get_stats(path,bwpath,top_idx,i)
                #spath = save_dir + "deltan" + "_" + str(i).zfill(4) + ".png"
                #csv_to_png(path,bwpath,spath,top_idx)
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
    work_dirs =  ["3%_crysta/","5%_crysta/","7%_crysta/"]
    
    # cavity dinamics +  delta_n
    volumes, cavityDepth, cavityMaxWidth , avgDeltan  = main_calculation(main_dir, work_dirs)
    
    create_plots(volumes, r'$V (\rm mm^3)$ ', "volumes", (0,2.5))
    create_plots(cavityDepth, r'$ D (\rm mm)$ ', "depth", (0,3))
    create_plots(cavityMaxWidth, r'$W_{\rm max} (\rm mm)$ ', "width", (0,1.5))
    create_plots(avgDeltan, r'$\Delta n_{\rm avg} (\rm nm)$ ', "deltan", (0,50))
