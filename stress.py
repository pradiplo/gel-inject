
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('classic')
import numpy as np
import re
import cv2
from os import listdir
from mpl_toolkits.axes_grid1 import make_axes_locatable

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def binarize(img):
    level = np.max(img)/10
    bw_img = np.where(img>level, 1, 0)
    bw_img = 1- bw_img
    return bw_img.astype("uint8")

def get_cav(path,ori_shape,bound):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #print(img.shape)
    img = cv2.resize(img,ori_shape[::-1],interpolation = cv2.INTER_AREA)
    img_dn = cv2.fastNlMeansDenoising(img)
    img_bin = binarize(img_dn)
    img_bin = img_bin[bound[0]:bound[1],bound[2]:bound[3]]
    #plt.imshow(img_bin,cmap="gray")
    #plt.show()
    black_idx = np.where(img_bin==1)
    #print(black_idx)
    return black_idx

def get_top_bound(path,dnpath):
    deltan = np.loadtxt(dnpath,delimiter=",")
    ori_shape =deltan.shape
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,ori_shape[::-1],interpolation = cv2.INTER_AREA)
    img_dn = cv2.fastNlMeansDenoising(img)
    start_idx = 90
    img_dn = img_dn[start_idx:-1,0:96]
    edges = cv2.Canny(img_dn,50,50)
    #contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #img_bin = binarize(img_dn)
    #print(contours)
    #cv2.drawContours(img_dn, contours, -1, (0,255,0), 3)
    #cv2.imshow('Contours', i)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(a)
    whites_idx= np.where(edges>1)
    top_idx = start_idx + min(whites_idx[0]) + 50
    #plt.imshow(whites,cmap="gray")
    #plt.show()
    #top_idx=0
    #print(top_idx)
    return top_idx

#dnpath ="/Users/tagawayo-lab-pc43/workspace/tagawalab/amed/raw-0420/test6_deltan/raw/CSV_retardation-00000001.csv"
#bwpath ="/Users/tagawayo-lab-pc43/workspace/tagawalab/amed/raw-0420/test6_deltan/bw/target000001.tif"
#deltan = np.loadtxt(dnpath,delimiter=",")
#orishape = deltan.shape
#bound = [160,440,0,96] #11
#bound = [232,512,0,96] #3
#bound = [110,390,0,96] #6
#deltan =deltan[bound[0]:bound[1],bound[2]:bound[3]]

#get_top_bound(bwpath,orishape)

#id = get_cav(bwpath,orishape,bound)
#deltan[id] = 0
#plt.imshow(deltan,vmin=0,vmax=120,cmap="viridis")
#plt.show()

def csv_to_png(path,bwpath,savepath,top_idx):   
    deltan = np.loadtxt(path,delimiter=",")
    ori_size = deltan.shape
    #top_idx = get_top_bound(bwpath,ori_size)
    #print(top_idx)
    bound = [top_idx, top_idx+280, 0, 96]
    deltan = deltan[bound[0]:bound[1],bound[2]:bound[3]]
    scale = 1.3100436681222708e-02
    mask_idx = get_cav(bwpath, ori_size,bound)
    deltan[mask_idx] = 0
    #deltan = cv2.fastNlMeansDenoisingColored(deltan)
    plt.rcParams['figure.figsize'] = (8, 12)
    plt.figure()
    ax = plt.subplot()
    im = ax.imshow(deltan,vmin=0,vmax=120,cmap="viridis",extent=[-48*scale,48*scale,-280*scale,0],aspect="auto")
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

def get_depth_width(path,ori_shape,bound):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #print(img.shape)
    img = cv2.resize(img,ori_shape[::-1],interpolation = cv2.INTER_AREA)
    img_dn = cv2.fastNlMeansDenoising(img)
    img_bin = binarize(img_dn)
    img_bin = img_bin[bound[0]:bound[1],bound[2]:bound[3]]
    #plt.imshow(img_bin,cmap="gray")
    #plt.show()
    whites = np.where(img_bin > 0)
  
    if len(whites[0]) > 0:
        depth = max(whites[0])
        width = max(whites[1]) - min(whites[1])
    else:
        depth = 0
        width = 0


    return depth, width

def get_stats(path,bwpath,savepath,top_idx):   
    deltan = np.loadtxt(path,delimiter=",")
    ori_size = deltan.shape
    #top_idx = get_top_bound(bwpath,ori_size)
    #print(top_idx)
    bound = [top_idx, top_idx+280, 0, 96]
    deltan = deltan[bound[0]:bound[1],bound[2]:bound[3]]
    scale = 1.3100436681222708e-02
    mask_idx = get_cav(bwpath, ori_size,bound)
    depth, width = get_depth_width(bwpath, ori_size,bound)
    deltan[mask_idx] = 0
    #deltan = cv2.fastNlMeansDenoisingColored(deltan)
    deln_avg = np.mean(deltan)
    vol = 0.25*np.pi * np.power((width*scale),2)* (depth*scale)
    return depth*scale, width*scale, deln_avg, vol

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
        print(bwpath + b)
        idx_mask = get_cav(bwpath + b,ori_shape,bound)
        #deltan[idx_mask] = 0    
        #if clean == True:
        #    deltan[0:170,50:60] = 0
        #    deltan = np.where(deltan > 80, 0, deltan)      
        avg = np.mean(deltan)
        deln.append(avg)
    time = np.linspace(0,len(deln),len(deln))*deltat*1000   
    return time, deln

main_dir = "/Users/tagawayo-lab-pc43/workspace/tagawalab/amed/gel-inject/data/"

work_dirs = ["test3","test12","test31","test32"]

#bound3 = [232,512,0,96]
#bound6 = [110,390,0,96]
#bound11 = [160,440,0,96]
#test3 -> 232:512
#test6 -> 110:390
#test11 -> 160:440
ddepths = []
wwidths = []
ddelns = []




for work_dir in work_dirs:

    #print(work_dir)
    
    raw_dir = main_dir + work_dir + "/CSVs/"
    save_dir = main_dir + "/movie/"
    bw_dir =main_dir + work_dir + "/"

    files = [i for i in sorted_nicely(listdir(raw_dir)) if i.endswith(".csv") and "axis" not in i ]
    #print(files)

    bwfiles = [i for i in sorted_nicely(listdir(bw_dir)) if i.endswith(".tif")]

    depths = []
    widths = []
    delns  = []
    vols = []
    
    for i in range(0,60):

        path = raw_dir + files[i]
        bwpath = bw_dir + bwfiles[i]
        spath = save_dir + work_dir + "_" + str(i).zfill(4) + ".png"
        
        print(spath)
        
        top_idx = get_top_bound(bw_dir + bwfiles[0],raw_dir + files[0])
        depth, width, deln_avg,vol = get_stats(path,bwpath,spath,top_idx)
        #csv_to_png(path,bwpath,spath,top_idx)
        depths.append(depth)
        widths.append(width)
        delns.append(deln_avg)
        vols.append(vol)

    deltat = 1/25000
    time = np.linspace(0,len(delns),len(delns))*deltat*1000   
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.figure()
    plt.plot(time, vols)
    plt.xlabel(r'$t$ (ms)',fontsize=40)
    plt.ylabel(r'$V (\rm mm^3)$ ',fontsize=40)
    plt.xlim(0,2.)
    #plt.ylim(0,40)
    plt.tick_params(axis='both', which='major', labelsize=32)
    plt.tick_params(axis='both', which='minor', labelsize=32)
    plt.tight_layout()
    plt.savefig(main_dir +work_dir + "vol.png")

    ddepths.append(depths)
    wwidths.append(widths)
    ddelns.append(delns)


width_stack = np.column_stack((wwidths[0],wwidths[1],wwidths[2]))
width_avg = np.mean(width_stack,axis=1)
width_std = np.std(width_stack,axis=1)


depth_stack = np.column_stack((ddepths[0],wwidths[1],wwidths[2]))
depth_avg = np.mean(depth_stack,axis=1)
depth_std = np.std(depth_stack,axis=1)

deln_stack = np.column_stack((ddelns[0],ddelns[1],ddelns[2]))
deln_avg = np.mean(deln_stack,axis=1)
deln_std = np.std(deln_stack,axis=1)

plt.rcParams['figure.figsize'] = (10, 8)
plt.figure()
plt.errorbar(time, width_avg,yerr=width_std)
plt.xlabel(r'$t$ (ms)',fontsize=40)
plt.ylabel(r'$w_{\rm max}$ (mm) ',fontsize=40)
plt.xlim(0,2.)
#plt.ylim(0,40)
plt.tick_params(axis='both', which='major', labelsize=32)
plt.tick_params(axis='both', which='minor', labelsize=32)
plt.tight_layout()
plt.savefig(main_dir + "width-ensemble.png")

plt.rcParams['figure.figsize'] = (10, 8)
plt.figure()
plt.errorbar(time, depth_avg,yerr=depth_std)
plt.xlabel(r'$t$ (ms)',fontsize=40)
plt.ylabel(r'$D$ (mm) ',fontsize=40)
plt.xlim(0,2.)
#plt.ylim(0,40)
plt.tick_params(axis='both', which='major', labelsize=32)
plt.tick_params(axis='both', which='minor', labelsize=32)
plt.tight_layout()
plt.savefig(main_dir + "depth-ensemble.png")

plt.rcParams['figure.figsize'] = (10, 8)
plt.figure()
plt.errorbar(time, deln_avg,yerr=deln_std)
plt.xlabel(r'$t$ (ms)',fontsize=40)
plt.ylabel(r'$\Delta n_{\rm avg}$ (nm) ',fontsize=40)
plt.xlim(0,2.)
#plt.ylim(0,40)
plt.tick_params(axis='both', which='major', labelsize=32)
plt.tick_params(axis='both', which='minor', labelsize=32)
plt.tight_layout()
plt.savefig(main_dir + "deln-ensemble.png")

"""
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.figure()
    plt.plot(time, depths)
    plt.xlabel(r'$t$ (ms)',fontsize=40)
    plt.ylabel(r'$D$ (mm) ',fontsize=40)
    plt.xlim(0,2.)
    #plt.ylim(0,40)
    plt.tick_params(axis='both', which='major', labelsize=32)
    plt.tick_params(axis='both', which='minor', labelsize=32)
    plt.tight_layout()
    plt.savefig("depth.png")

    plt.rcParams['figure.figsize'] = (10, 8)
    plt.figure()
    plt.plot(time, widths)
    plt.xlabel(r'$t$ (ms)',fontsize=40)
    plt.ylabel(r'$W_{\rm max}$ (mm) ',fontsize=40)
    plt.xlim(0,2.)
    #plt.ylim(0,40)
    plt.tick_params(axis='both', which='major', labelsize=32)
    plt.tick_params(axis='both', which='minor', labelsize=32)
    plt.tight_layout()
    plt.savefig("width.png")


    plt.rcParams['figure.figsize'] = (10, 8)
    plt.figure()
    plt.plot(time, delns)
    plt.xlabel(r'$t$ (ms)',fontsize=40)
    plt.ylabel(r'$\Delta n_{\rm avg}$ (nm) ',fontsize=40)
    plt.xlim(0,2.)
    #plt.ylim(0,40)
    plt.tick_params(axis='both', which='major', labelsize=32)
    plt.tick_params(axis='both', which='minor', labelsize=32)
    plt.tight_layout()
    plt.savefig("deltan.png")
    """



"""
plt.rcParams['figure.figsize'] = (10, 8)
plt.figure()
plt.plot(time, depths)
plt.xlabel(r'$t$ (ms)',fontsize=40)
plt.ylabel(r'$D$ (mm) ',fontsize=40)
plt.xlim(0,2.)
#plt.ylim(0,40)
plt.tick_params(axis='both', which='major', labelsize=32)
plt.tick_params(axis='both', which='minor', labelsize=32)
plt.tight_layout()
 plt.savefig("depth.png")

plt.rcParams['figure.figsize'] = (10, 8)
plt.figure()
plt.plot(time, widths)
plt.xlabel(r'$t$ (ms)',fontsize=40)
plt.ylabel(r'$W_{\rm max}$ (mm) ',fontsize=40)
plt.xlim(0,2.)
#plt.ylim(0,40)
plt.tick_params(axis='both', which='major', labelsize=32)
plt.tick_params(axis='both', which='minor', labelsize=32)
plt.tight_layout()
plt.savefig("width.png")


plt.rcParams['figure.figsize'] = (10, 8)
plt.figure()
plt.plot(time, delns)
plt.xlabel(r'$t$ (ms)',fontsize=40)
plt.ylabel(r'$\Delta n_{\rm avg}$ (nm) ',fontsize=40)
plt.xlim(0,2.)
#plt.ylim(0,40)
plt.tick_params(axis='both', which='major', labelsize=32)
plt.tick_params(axis='both', which='minor', labelsize=32)
plt.tight_layout()
plt.savefig("deltan.png")
"""


"""
wd3 = main_dir + "test3_deltan"
wd6 = main_dir +"test6_deltan"
wd11 =main_dir + "test11_deltan"



dn3 = get_spatial_avg(wd3 + "/raw/", wd3 + "/bw/" ,232,512)
dn6 = get_spatial_avg(wd6 + "/raw/", wd6 + "/bw/" ,110,390)
dn11 = get_spatial_avg(wd11 + "/raw/", wd11 + "/bw/",160,440)
 
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

plt.rcParams['figure.figsize'] = (10, 8)
plt.figure()
plt.plot(dn11[0],dn11[1],"r",linewidth=2,label="3w.t%")
plt.plot(dn3[0],dn3[1],"g",linewidth=2,label="5w.t%")
plt.plot(dn6[0],dn6[1],"b",linewidth=2,label="7w.t%")
plt.xlabel(r'$t$ (ms)',fontsize=40)
plt.ylabel(r'$\Delta n_{\rm avg}$ (nm) ',fontsize=40)
plt.xlim(0,2.7)
plt.ylim(0,40)
plt.tick_params(axis='both', which='major', labelsize=32)
plt.tick_params(axis='both', which='minor', labelsize=32)
plt.legend(loc="best",prop={'size': 30}, framealpha=0.5)
plt.tight_layout()
#plt.show()
plt.savefig("delnavg.png")
"""