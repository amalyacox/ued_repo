#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
from UED_Data_Reduction import * 
from UED_Diffuse_Analysis import * 
sys.path.append('/cds/group/ued/scratch/jupyter_notebook_UED_solid_sample_codes/python_packages/')
from ued_solid_state_FY19_utilities import *
sys.path.append('/cds/group/ued/scratch/jupyter_notebook_UED_solid_sample_codes/python_packages/polarTransform-2.0.0/')
from gas_phase_UED_cython_functions import *


def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = (angle_range)#np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def gen_circular_1pix_mask(x0, y0, out_r, num_row, num_col):
    in_r = 1 - out_r
    mask = np.zeros((num_row, num_col))
    
    X,Y = np.meshgrid(np.arange(num_col), np.arange(num_row))
    mask[(X-x0)**2+(Y-y0)**2 < out_r**2] = 1
    mask[(X-x0)**2+(Y-y0)**2 < in_r**2] = 0
    return mask

def inten_r_circ(peak, img, r):
    y, x = peak
    if type(r) == int:  
        mask = gen_circular_1pix_mask(x,y,r, img.shape[0] ,img.shape[1])
        fac = len(np.nonzero(mask)[0]) + len(np.nonzero(mask)[1])
        inten = np.nansum(img*mask) / fac
        return inten 
    elif type(r) == np.ndarray: 
        inten_arr = []
        for rad in r: 
            mask = gen_circular_1pix_mask(x,y,rad, img.shape[0] ,img.shape[1])
            fac = len(np.nonzero(mask)[0]) + len(np.nonzero(mask)[1])
            inten = np.nansum(img*mask) / fac
            inten_arr.append(inten)
        return inten_arr 
        
def find_t(peak):
    I1 = avg_img_mean_set[-5:,:,:].mean(axis=0)
    ind = np.argwhere(delay_time == 0)[0][0]
    j = ind - 5 
    while j > 41: 
        I3 = avg_img_mean_set[j,:,:]
        delI = I3 - I1
        delI_p = delI/I1*100
        inten = inten_r_circ(peak, delI_p, 1)
        if inten < -1: 
            break 
        else: 
            j-=1
    return j

def find_r(bragginfo):
    peaks = bragginfo['centroids_bragg_all']
    I1 = avg_img_mean_set[-5:,:,:].mean(axis=0)
    r_vals = []
    for peak in tqdm.tqdm(peaks): 
        j = find_t(peak)
        I3 = avg_img_mean_set[j,:,:]
        delI = I3 - I1
        delI_p = delI/I1*100
        r = 7
        while r < 30: 
            inten = inten_r_circ(peak, delI_p, r)
            if inten > 0: 
                r_vals.append((j, r)) 
                break 
            else: 
                r+=1 
    return r_vals

def get_avg_img(img, radius=70):
    """
    Generate a 60 degree arc of averaged data from one img 
    inputs: img(np.ndarray) one diffraction image, 
            radius(int): default 70, how far out do we want to go (toggles how many bragg peaks we're including)
    outputs: avg_img (np.ndarray): same dimensions as input image, but only nonzero areas are the averaged pixels
    """
    global centroids_bragg
    global centroid_main_beam
    centroids_bragg = sort_bragg_peaks_centroids(centroids_bragg, centroid_main_beam)
    vec_a1 = (centroids_bragg[1,:] - centroid_main_beam)/1.99

    theta = -60/180*np.pi
    rot_M = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    vec_b1 = vec_a1.dot(rot_M)
    t0 = np.arctan(vec_a1[1]/vec_a1[0])    
    
    avg_img = np.zeros(img.shape)

    inner = (1-gen_circular_mask(centroid_main_beam[1], centroid_main_beam[0], radius, img.shape[0], img.shape[1]))
    arc = sector_mask((img.shape[0], img.shape[1]), centroid_main_beam, 500, (t0,t0-theta))
    for ind in tqdm.tqdm(np.argwhere(img*inner*arc)):
        ij_list = []
        for k in [1,2,3,4,5]:
            pos = (ind[0], ind[1])
            vec = (ind - centroid_main_beam)/1.99
            t = np.arctan(vec[1]/vec[0])
            t_new = t + -60/180 * np.pi * k
            pos_new = centroid_main_beam + la.norm(pos - centroid_main_beam)*np.array([np.cos(t_new), np.sin(t_new)])
            inew, jnew = int(pos_new[0]), int(pos_new[1])
            ij_list.append((inew,jnew))
        avg_img[pos] = np.nanmean([img[loc] for loc in ij_list]) 
    return avg_img 

def get_avg_img_set(radius=70):
    """
    Generate average image for each image in rescaled image set 
    This takes a long time. Need to tweek somehow
    Inputs: radius(int): default  70
            imgset(np.ndarray): default: img_set_rescaled (set of rescaled diffraction images at different time delays)
    """
    global img_set_rescaled
    imgset = img_set_rescaled
    avg_img_set = []
    for ind1 in tnrange(imgset.shape[0]):
        temp = imgset[ind1,:,:] 
        avg_img = get_avg_img(temp, radius)
        avg_img_set.append(avg_img)
    avg_img_set = np.array(avg_img_set)
    return avg_img_set

def get_avg_img_mean_set(radius=70):
    """
    With the average image set, average each image with images of the same delay time
    Initially looks for avg_img_set to be loaded
    inputs: radius (int)
    outputs: avg_img_mean_set (np.ndarray), shape: [len(delay_time), avg_img[0], avg_img[1]]
    """
    delay_unique = np.unique(delay)
    global avg_img_set 
        
    try:
        avg_img_mean_set = np.zeros((len(delay_unique), avg_img_set.shape[1], avg_img_set.shape[2]))
        for ind in tnrange(len(delay_unique)):
            ind_de = np.where(delay==delay_unique[ind])[0]
            avg_img_mean_set[ind,:,:] = np.mean(avg_img_set[ind_de,:,:], axis=0)
            
        return avg_img_mean_set
    
    except NameError: 
        global img_set_rescaled
        imgset = img_set_rescaled
    
        avg_img_set = get_avg_img_set(radius)
        avg_img_mean_set = np.zeros((len(delay_unique), avg_img_set.shape[1], avg_img_set.shape[2]))
        
        for ind in tnrange(len(delay_unique)):
            ind_de = np.where(delay==delay_unique[ind])[0]
            avg_img_mean_set[ind,:,:] = np.mean(avg_img_set[ind_de,:,:], axis=0)
        
        return avg_img_mean_set