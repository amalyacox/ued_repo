#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as curve_fit
import os

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def find_name(filename):
    for i in range(len(filename)):
        if filename[i:i+4] == 'data':
            name = filename[i+13:]
            break 
    for i in range(len(name)): 
        if name[i:i+2] == 'h5':
            name = name[:i-1]
            break
    return name


def material(filename):
    """
    Return the material of a specific bragg_info dictionary
    inputs: (str) path of bragg info dict
    returns: str MoSe2/WSe2/MoS2/WS2 
    """
    mats = ['MoSe2', 'WSe2', 'MoS2', 'WS2']
    for mat in mats: 
        if mat in filename: 
            return mat

def Q(order, a):
    """
    Length of reciprocal lattice vector to specific order 
    inputs:order, lattice parameter (angstrom)
    outputs: Q #angstrom inverse
    Lattice constants defined from Zhuang, H., et. al, J. Phys. Chem. C 2013, 117, 40, 20440–20445
    """
    a_dict = {'MoSe2': 3.32, 'WSe2': 3.32, 'MoS2':3.18, 'WS2':3.18}
    bo_dict = {'order1' : [1,0,0], 'order2' : [2,-1,0], 
           'order3' : [2,0,0], 'order4' :[3,-1,0], 
           'order5' : [3,0,0], 'order6' : [4,-2,0], 
           'order7' : [4,-1,0], 'order8' : [4,0,0], 
           'order9' : [5,-2,0], 'order10' : [5,-1,0]}
           
    bo = bo_dict['order'+str(order)]
    return (2*np.pi * (np.sqrt(4/(3*a**2) * (bo[0]**2 + bo[0]*bo[1] + bo[1]**2))))

def linear(x,m):
    """
    Linear function for fitting u_rms, b = 0, line goes through 0 
    """ 
    b=0
    return x*m + b

def rise_decay(x, y0, A, x0, tau_1, tau_2): 
    """
    Exponential rise & decay convolution with sigma = 0.15; instrument resolution 
    Inputs: x, y0, A, x0, tau_1, tau_2. x: independent paramater for fit 
    Outputs: exponential rise & decay curve
    """
    return y0 + A*(-np.exp(-(x-x0)/tau_1 + (0.15**2)/(2*tau_1**2)) + np.exp(-(x-x0)/tau_2 + (0.15**2)/(2*tau_2**2)))

# def rise_rise_decay(x, y0, A, x0, tau_1, tau_2, tau_3,a): 
#     """
#     Exponential rise & decay convolution with sigma = 0.15; instrument resolution 
#     Inputs: x, y0, A, x0, tau_1, tau_2. x: independent paramater for fit 
#     Outputs: exponential rise & decay curve
#     """
#     return y0 + A*(-(1-a)*np.exp(-(x-x0)/tau_1 + (0.15**2)/(2*tau_1**2)) + -a*np.exp(-(x-x0)/tau_3 + (0.15**2)/(2*tau_3**2)) + np.exp(-(x-x0)/tau_2 + (0.15**2)/(2*tau_2**2)))

def dblexppeak(x, y0, A, x0, tau_1, tau_2):
    """
    Exponential rise and decay without including instrument resolution param (sigma). Fit with this function first, 
    then use popt for fitting rise_decay 
    Inputs: x, y0, A, x0, tau_1, tau_2. x: independent parameter for fit. 
    Output: exponential rise & decay curve
    """
    return y0 + A*(-np.exp(-(x-x0)/tau_1) + np.exp((-x-x0)/tau_2))

def tau1_info1(fn):
    """
    inputs: scan directory (str)
    outputs: tau1 of bragginfo1 
    """
    fit_params = pd.read_csv(f'{fn}/bragginfo1_fitparams.txt')
    return fit_params['val'][3], fit_params['err'][3]
    
def tau1_info2(fn):
    """
    inputs: scan directory (str)
    outputs: tau1 of bragginfo2
    """
    fit_params = pd.read_csv(f'{fn}/bragginfo2_fitparams.txt')
    return fit_params['val'][3], fit_params['err'][3]

def tau2_info1(fn):
    """
    inputs: scan directory (str)
    outputs: tau2 of bragginfo1
    """
    fit_params = pd.read_csv(f'{fn}/bragginfo1_fitparams.txt')
    return fit_params['val'][4], fit_params['err'][4]

def tau2_info2(fn):
    """
    inputs: scan directory (str)
    outputs: tau2 of bragginfo2
    """
    fit_params = pd.read_csv(f'{fn}/bragginfo2_fitparams.txt')
    return fit_params['val'][4], fit_params['err'][4]

def A_info1(fn):
    """
    inputs: scan directory (str)
    outputs:  of bragginfo2
    """
    fit_params = pd.read_csv(f'{fn}/bragginfo1_fitparams.txt')
    return fit_params['val'][1], fit_params['err'][1]

def A_info2(fn):
    fit_params = pd.read_csv(f'{fn}/bragginfo2_fitparams.txt')
    return fit_params['val'][1], fit_params['err'][1]

def get_fn(material, fluence, temp, pump, deg, data_type):
    path = '/home/amalyajohnson/Research/UED_U085/Processed_Data/'
    for file in os.listdir(path):
        if file.startswith('2021'):
            data_path = os.path.join(path, file)
            for i in os.listdir(data_path):
                if '_bragginfo1' in i: 
                    data = scan(data_path)
                    if data.fluence == fluence and data.temp == temp and data.pump==pump:
                        if data_type == 'ML' and data.type == data_type and data.bragg_info1.mat == material:
                            print(data_path)
                            return data_path 
                        elif data_type == 'HS' and data.type == data_type and data.deg == deg:
                            if data.bragg_info1.mat == material or data.bragg_info2.mat == material:
                                print(data_path)
                                return data_path 
                                

def get_tau(x,y,err):
    """
    Inputs: x, y, err: array-like: delay time, average intensity, and error
    Outputs: fit, fit_up, fit_down, t1, t2, perr, popt, result
    """
    nstd = 2 # to draw 2-sigma intervals
    p0, pcov_init = curve_fit(dblexppeak, x, y, sigma = err) 
    popt, pcov = curve_fit(rise_decay, x, y, sigma=err, p0=p0)
    
    if popt[-1] < popt[-2]:
            p0, pcov = curve_fit(dblexppeak, x,y, sigma=err, bounds =([-np.inf, -np.inf, -np.inf, 0,0], [np.inf,0,np.inf,np.inf,np.inf]))
            try: 
                popt, pcov = curve_fit(rise_decay, x, y, sigma=err, p0=p0)
                perr = np.sqrt(np.diag(pcov))

                t1  = popt[-2]
                t2 = popt[-1]
                if t2 < t1: 
                    popt[-1] = t1
                    popt[-2] = t2
                    popt, pcov = curve_fit(rise_decay, x, y, sigma=err, p0=popt)
                    perr = np.sqrt(np.diag(pcov))
                    t1  = popt[-2]
                    t2 = popt[-1]
                    print('made it here')
                popt_up = popt + nstd * perr
                popt_dw = popt - nstd * perr

                fit = rise_decay(x, *popt)
                fit_up = rise_decay(x, *popt_up)
                fit_dw = rise_decay(x, *popt_dw)
                result = 'Fit with exponential rise & decay convolution'
            except RuntimeError:
                perr = np.sqrt(np.diag(pcov))
                t1  = popt[-2]
                t2 = popt[-1]
                popt_up = p0 + nstd * perr
                popt_dw = p0 - nstd * perr
                fit = rise_decay(x, *popt)
                fit_up = rise_decay(x, *popt_up)
                fit_dw = rise_decay(x, *popt_dw)
                result = 'Fit failed, fit with simple exponential rise & decay'
    else: 
        perr = np.sqrt(np.diag(pcov))

        t1  = popt[-2]
        t2 = popt[-1]
        popt_up = popt + nstd * perr
        popt_dw = popt - nstd * perr

        fit = rise_decay(x, *popt)
        fit_up = rise_decay(x, *popt_up)
        fit_dw = rise_decay(x, *popt_dw)
        result = 'Fit with exponential rise & decay convolution'
    return fit, fit_up, fit_dw, t1, t2, perr, popt, result

def plot_log(material, fluence, temp, pump, deg, order_num=10, include_ML=True): #pump
    path = '/cds/home/a/amalyaj/Data/post_alignment/'
    for file in os.listdir(path):
        if file.startswith('2021'):
            data_path = os.path.join(path, file)
            for i in os.listdir(data_path):
                if 'bragginfo1' in i:
                    data = scan(data_path)
                    if data.fluence == fluence and data.temp == temp and data.pump == pump: 
                        if data.deg == deg: 
                            if data.bragg_info1.mat == material:
                                data.fit_log(order_num)
                                plt.scatter(data.data1['q_sq'], -data.data1['log_inten_1'], label=data.bragg_info1.mat+'_'+str(data.deg)+'deg_'+data.temp+'_'+data.fluence)
                                plt.plot(data.data1['q_sq'], data.data1['log_fit1'])
#                                 plt.fill_between(data.data1['q_sq'], data.data1['log_fit1_up'], data.data1['log_fit1_dw'], alpha=0.25)
#                                 print(data_path)
                            elif data.bragg_info2.mat == material:
                                data.fit_log(order_num)
                                plt.scatter(data.data2['q_sq'], -data.data2['log_inten_2'], label=data.bragg_info2.mat+'_'+str(data.deg)+'deg_'+data.temp+'_'+data.fluence)
                                plt.plot(data.data2['q_sq'], data.data2['log_fit2'])
#                                 plt.fill_between(data.data2['q_sq'], data.data2['log_fit2_up'], data.data2['log_fit2_dw'], alpha=0.25)
#                                 print(data_path)
                        elif data.bragg_info1.mat == material and data.type == 'ML' and include_ML:
                            data.fit_log(order_num)
                            plt.scatter(data.data1['q_sq'], -data.data1['log_inten_1'], label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                            plt.plot(data.data1['q_sq'], data.data1['log_fit1'])
                            # plt.fill_between(data.data1['q_sq'], data.data1['log_fit1_up'], data.data1['log_fit1_dw'], alpha=0.25)
#                             print(data_path)

# def plot_rms(material, fluence, temp, pump, deg, order_num=10, include_ML=True):
#     path = '/cds/home/a/amalyaj/Data/post_alignment/'
# #     plt.figure()
#     for file in os.listdir(path):
#         if file.startswith('2021'):
#             data_path = os.path.join(path, file)
#             for i in os.listdir(data_path):
#                 if 'bragginfo1' in i:
#                     data = scan(data_path)
#                     if data.fluence == fluence and data.temp == temp and data.pump == pump:
#                         if data.deg == deg: 
#                             if data.bragg_info1.mat == material:
#                                 data.rms(order_num)
#                                 plt.errorbar(data.delay, data.rms1, data.rms1_err, label=data.bragg_info1.mat+'_'+str(data.deg)+'deg_'+data.temp+'_'+data.fluence)
#                                 print(data_path)
#                             elif data.bragg_info2.mat == material:
#                                 data.rms(order_num)
#                                 plt.errorbar(data.delay, data.rms2, data.rms2_err, label=data.bragg_info2.mat+'_'+str(data.deg)+'deg_'+data.temp+'_'+data.fluence)
#                                 print(data_path)
#                         elif data.bragg_info1.mat == material and data.type == 'ML' and include_ML:
#                             data.rms(order_num)
#                             plt.errorbar(data.delay, data.rms1, data.rms1_err, label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
#                             print(data_path)
def plot_tau1_fluence(material, fluence, temp, pump, deg, order_num=10, include_ML=True): 
    path = '/cds/home/a/amalyaj/Data/post_alignment/'
#     plt.figure()
    for file in os.listdir(path):
        if file.startswith('2021') and '0011' not in file:
            data_path = os.path.join(path, file)
            for i in os.listdir(data_path):
                if 'bragginfo1_fitparams.txt' in i: 
                    data = scan(data_path)
                    if data.fluence == fluence and data.temp == temp and data.pump == pump:
                        pwr = float(data.fluence.replace('mj',''))
                        if data.deg == deg: 
                            if data.bragg_info1.mat == material and data.type == 'HS':
                                if data.bragg_info1.mat == 'MoSe2' or data.bragg_info1.mat == 'MoS2':
                                    plt.scatter(pwr, tau1_info1(data_path)[0], color='k', label=data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(pwr, tau1_info1(data_path)[0], tau1_info1(data_path)[1], color='k')
                                else:
                                    plt.scatter(pwr+0.1, tau1_info1(data_path)[0], color='r', label=data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(pwr+0.1, tau1_info1(data_path)[0], tau1_info1(data_path)[1], color='r')
                            elif data.bragg_info2.mat == material:
                                if data.bragg_info2.mat == 'MoSe2' or data.bragg_info2.mat == 'MoS2':
                                    plt.scatter(pwr, tau1_info2(data_path)[0], color='k', label=data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(pwr, tau1_info2(data_path)[0], tau1_info2(data_path)[1], color='k')
                                else:
                                    plt.scatter(pwr+0.1, tau1_info2(data_path)[0], color='r', label=data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(pwr+0.1, tau1_info2(data_path)[0], tau1_info2(data_path)[1], color='r')
                                
                        elif data.bragg_info1.mat == material and data.type == 'ML' and include_ML:
                            if data.bragg_info1.mat == 'MoSe2' or data.bragg_info1.mat == 'MoS2':
                                plt.scatter(pwr, tau1_info1(data_path)[0], color='k', marker='*',label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                                plt.errorbar(pwr, tau1_info1(data_path)[0], tau1_info1(data_path)[1], color='k')
                            else:
                                plt.scatter(pwr+0.1, tau1_info1(data_path)[0], color='r', marker='*', label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                                plt.errorbar(pwr+0.1, tau1_info1(data_path)[0], tau1_info1(data_path)[1], color='r')
                                
                                
def plot_tau2_fluence(material, fluence, temp, pump, deg, order_num=10, include_ML=True): 
    path = '/cds/home/a/amalyaj/Data/post_alignment/'
#     plt.figure()
    for file in os.listdir(path):
        if file.startswith('2021') and '0011' not in file:
            data_path = os.path.join(path, file)
            for i in os.listdir(data_path):
                if 'bragginfo1_fitparams.txt' in i: 
                    data = scan(data_path)
                    if data.fluence == fluence and data.temp == temp and data.pump == pump:
                        pwr = float(data.fluence.replace('mj',''))
                        if data.deg == deg: 
                            if data.bragg_info1.mat == material and data.type == 'HS':
                                if data.bragg_info1.mat == 'MoSe2' or data.bragg_info1.mat == 'MoS2':
                                    plt.scatter(pwr, tau2_info1(data_path)[0], color='k', label=data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(pwr, tau2_info1(data_path)[0], tau2_info1(data_path)[1], color='k')
                                else:
                                    plt.scatter(pwr+0.1, tau2_info1(data_path)[0], color='r', label=data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(pwr+0.1, tau2_info1(data_path)[0], tau2_info1(data_path)[1], color='r')
                            elif data.bragg_info2.mat == material:
                                if data.bragg_info2.mat == 'MoSe2' or data.bragg_info2.mat == 'MoS2':
                                    plt.scatter(pwr, tau2_info2(data_path)[0], color='k', label=data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(pwr, tau2_info2(data_path)[0], tau2_info2(data_path)[1], color='k')
                                else:
                                    plt.scatter(pwr+0.1, tau2_info2(data_path)[0], color='r', label=data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(pwr+0.1, tau2_info2(data_path)[0], tau2_info2(data_path)[1], color='r')
                                
                        elif data.bragg_info1.mat == material and data.type == 'ML' and include_ML:
                            print('here')
                            if data.bragg_info1.mat == 'MoSe2' or data.bragg_info1.mat == 'MoS2':
                                plt.scatter(pwr, tau2_info1(data_path)[0], color='k', marker='*',label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                                plt.errorbar(pwr, tau2_info1(data_path)[0], tau2_info1(data_path)[1], color='k')
                            else:
                                plt.scatter(pwr+0.1, tau2_info1(data_path)[0], color='r', marker='*', label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                                plt.errorbar(pwr+0.1, tau2_info1(data_path)[0], tau2_info1(data_path)[1], color='r')

def plot_tau1_angle(material, fluence, temp, pump, deg, order_num=10, include_ML=True): 
    path = '/cds/home/a/amalyaj/Data/post_alignment/'
#     plt.figure()
    for file in os.listdir(path):
        if file.startswith('2021') and '0011' not in file:
            data_path = os.path.join(path, file)
            for i in os.listdir(data_path):
                if 'bragginfo1_fitparams.txt' in i: 
                    data = scan(data_path)
                    if data.fluence == fluence and data.temp == temp and data.pump == pump:
                        if data.deg == deg: 
                            if data.bragg_info1.mat == material:
                                if data.bragg_info1.mat == 'MoSe2' or data.bragg_info1.mat == 'MoS2':
                                    plt.scatter(data.deg, tau1_info1(data_path)[0], color='k', label=data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(data.deg, tau1_info1(data_path)[0], tau1_info1(data_path)[1], color='k')
                                else:
                                    plt.scatter(data.deg+0.1, tau1_info1(data_path)[0], color='r', label=data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(data.deg+0.1, tau1_info1(data_path)[0], tau1_info1(data_path)[1], color='r')
                            elif data.bragg_info2.mat == material:
                                if data.bragg_info2.mat == 'MoSe2' or data.bragg_info2.mat == 'MoS2':
                                    plt.scatter(data.deg, tau1_info2(data_path)[0], color='k', label=data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(data.deg, tau1_info2(data_path)[0], tau1_info2(data_path)[1], color='k')
                                else:
                                    plt.scatter(data.deg+0.1, tau1_info2(data_path)[0], color='r', label=data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(data.deg+0.1, tau1_info2(data_path)[0], tau1_info2(data_path)[1], color='r')
                                
                        elif data.bragg_info1.mat == material and data.type == 'ML' and include_ML:
                            if data.bragg_info1.mat == 'MoSe2' or data.bragg_info1.mat == 'MoS2':
                                plt.scatter(0, tau1_info1(data_path)[0], color='k', marker='*',label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                                plt.errorbar(0, tau1_info1(data_path)[0], tau1_info1(data_path)[1], color='k')
                            else:
                                plt.scatter(0.1, tau1_info1(data_path)[0], color='r', marker='*', label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                                plt.errorbar(0.1, tau1_info1(data_path)[0], tau1_info1(data_path)[1], color='r')

def plot_tau2_angle(material, fluence, temp, pump, deg, order_num=10, include_ML=True): 
    path = '/cds/home/a/amalyaj/Data/post_alignment/'
#     plt.figure()
    for file in os.listdir(path):
        if file.startswith('2021') and '0011' not in file:
            data_path = os.path.join(path, file)
            for i in os.listdir(data_path):
                if 'bragginfo1_fitparams.txt' in i: 
                    data = scan(data_path)
                    if data.fluence == fluence and data.temp == temp and data.pump == pump:
                        if data.deg == deg: 
                            if data.bragg_info1.mat == material:
                                if data.bragg_info1.mat == 'MoSe2' or data.bragg_info1.mat == 'MoS2':
                                    plt.scatter(data.deg, tau2_info1(data_path)[0], color='k', label=data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(data.deg, tau2_info1(data_path)[0], tau2_info1(data_path)[1], color='k')
                                else:
                                    plt.scatter(data.deg+0.1, tau2_info1(data_path)[0], color='r', label=data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(data.deg+0.1, tau2_info1(data_path)[0], tau2_info1(data_path)[1], color='r')
                            elif data.bragg_info2.mat == material:
                                if data.bragg_info2.mat == 'MoSe2' or data.bragg_info2.mat == 'MoS2':
                                    plt.scatter(data.deg, tau2_info2(data_path)[0], color='k', label=data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(data.deg, tau2_info2(data_path)[0], tau2_info2(data_path)[1], color='k')
                                else:
                                    plt.scatter(data.deg+0.1, tau2_info2(data_path)[0], color='r', label=data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(data.deg+0.1, tau2_info2(data_path)[0], tau2_info2(data_path)[1], color='r')
                                
                        elif data.bragg_info1.mat == material and data.type == 'ML' and include_ML:
                            if data.bragg_info1.mat == 'MoSe2' or data.bragg_info1.mat == 'MoS2':
                                plt.scatter(0, tau2_info1(data_path)[0], color='k', marker='*',label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                                plt.errorbar(0, tau2_info1(data_path)[0], tau2_info1(data_path)[1], color='k')
                            else:
                                plt.scatter(0.1, tau2_info1(data_path)[0], color='r', marker='*', label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                                plt.errorbar(0.1, tau2_info1(data_path)[0], tau2_info1(data_path)[1], color='r')
                                
def plot_A_angle(material, fluence, temp, pump, deg, order_num=10, include_ML=True): 
    path = '/cds/home/a/amalyaj/Data/post_alignment/'
#     plt.figure()
    for file in os.listdir(path):
        if file.startswith('2021') and '0011' not in file:
            data_path = os.path.join(path, file)
            for i in os.listdir(data_path):
                if 'bragginfo1_fitparams.txt' in i: 
                    data = scan(data_path)
                    if data.fluence == fluence and data.temp == temp and data.pump == pump:
                        if data.deg == deg: 
                            if data.bragg_info1.mat == material:
                                if data.bragg_info1.mat == 'MoSe2' or data.bragg_info1.mat == 'MoS2':
                                    plt.scatter(data.deg, A_info1(data_path)[0], color='k', label=data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(data.deg, A_info1(data_path)[0], A_info1(data_path)[1], color='k')
                                else:
                                    plt.scatter(data.deg+0.1, tau2_info1(data_path)[0], color='r', label=data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(data.deg+0.1, tau2_info1(data_path)[0], A_info1(data_path)[1], color='r')
                            elif data.bragg_info2.mat == material:
                                if data.bragg_info2.mat == 'MoSe2' or data.bragg_info2.mat == 'MoS2':
                                    plt.scatter(data.deg, A_info2(data_path)[0], color='k', label=data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(data.deg, A_info2(data_path)[0], A_info2(data_path)[1], color='k')
                                else:
                                    plt.scatter(data.deg+0.1, A_info2(data_path)[0], color='r', label=data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(data.deg+0.1, A_info2(data_path)[0], A_info2(data_path)[1], color='r')
                                
                        elif data.bragg_info1.mat == material and data.type == 'ML' and include_ML:
                            if data.bragg_info1.mat == 'MoSe2' or data.bragg_info1.mat == 'MoS2':
                                plt.scatter(0, A_info1(data_path)[0], color='k', marker='*',label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                                plt.errorbar(0, A_info1(data_path)[0], A_info1(data_path)[1], color='k')
                            else:
                                plt.scatter(0.1, A_info1(data_path)[0], color='r', marker='*', label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                                plt.errorbar(0.1, A_info1(data_path)[0], A_info1(data_path)[1], color='r')

def plot_A_fluence(material, fluence, temp, pump, deg, order_num=10, include_ML=True): 
    path = '/cds/home/a/amalyaj/Data/post_alignment/'
#     plt.figure()
    for file in os.listdir(path):
        if file.startswith('2021') and '0011' not in file:
            data_path = os.path.join(path, file)
            for i in os.listdir(data_path):
                if 'bragginfo1_fitparams.txt' in i: 
                    data = scan(data_path)
                    if data.fluence == fluence and data.temp == temp and data.pump == pump:
                        pwr = float(data.fluence.replace('mj',''))
                        if data.deg == deg: 
                            if data.bragg_info1.mat == material and data.type == 'HS':
                                if data.bragg_info1.mat == 'MoSe2' or data.bragg_info1.mat == 'MoS2':
                                    plt.scatter(pwr, A_info1(data_path)[0], color='k', label=data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(pwr, A_info1(data_path)[0], A_info1(data_path)[1], color='k')
                                else:
                                    plt.scatter(pwr+0.1, A_info1(data_path)[0], color='r', label=data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(pwr+0.1, A_info1(data_path)[0], A_info1(data_path)[1], color='r')
                            elif data.bragg_info2.mat == material:
                                if data.bragg_info2.mat == 'MoSe2' or data.bragg_info2.mat == 'MoS2':
                                    plt.scatter(pwr, A_info2(data_path)[0], color='k', label=data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(pwr, A_info2(data_path)[0], A_info2(data_path)[1], color='k')
                                else:
                                    plt.scatter(pwr+0.1, A_info2(data_path)[0], color='r', label=data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence)
                                    plt.errorbar(pwr+0.1, A_info2(data_path)[0], A_info2(data_path)[1], color='r')
                                
                        elif data.bragg_info1.mat == material and data.type == 'ML' and include_ML:
                            print('here')
                            if data.bragg_info1.mat == 'MoSe2' or data.bragg_info1.mat == 'MoS2':
                                plt.scatter(pwr, A_info1(data_path)[0], color='k', marker='*',label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                                plt.errorbar(pwr, A_info1(data_path)[0], A_info1(data_path)[1], color='k')
                            else:
                                plt.scatter(pwr+0.1, A_info1(data_path)[0], color='r', marker='*', label=data.bragg_info1.mat+'_'+data.type+'_'+data.temp+'_'+data.fluence)
                                plt.errorbar(pwr+0.1, A_info1(data_path)[0], A_info1(data_path)[1], color='r')


def plot_rms(material, fluence, temp, pump, deg, data_type):
    fn = get_fn(material, fluence, temp, pump, deg, data_type)
    data = scan(fn)
    data.rms()
    if data.bragg_info1.mat == material: 
        info = data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence
        plt.errorbar(data.delay, data.rms1, data.rms1_err, label=info )
        plt.xlabel('Delay time(ps)')
        plt.ylabel(r'$\Delta \langle u_{\perp}^2 \rangle [\AA^2]$')
    elif data_type == 'HS' and data.bragg_info2.mat == material:
        info = data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence
        plt.errorbar(data.delay, data.rms2, data.rms2_err, label=info)
        plt.xlabel('Delay time(ps)')
        plt.ylabel(r'$\Delta \langle u_{\perp}^2 \rangle [\AA^2]$')
    
    
    
def plot_rawdata(material, fluence, temp, pump, deg, data_type, order):
    fn = get_fn(material, fluence, temp, pump, deg, data_type)
    data = scan(fn)
    if data.bragg_info1.mat == material: 
        info = data.bragg_info1.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence+'_order'+str(order)
        plt.errorbar(data.delay, data.bragg_info1['bragg_inten_mean_order'+str(order)], data.bragg_info1['bragg_inten_err_order'+str(order)], label=info)
        plt.ylabel(rf'$\Delta I / I_0$')
        plt.xlabel('Delay (ps)')
    elif data_type == 'HS' and data.bragg_info2.mat == material:
        info = data.bragg_info2.mat+'_'+str(data.deg)+'_'+data.temp+'_'+data.fluence+'_order'+str(order)
        plt.errorbar(data.delay, data.bragg_info2['bragg_inten_mean_order'+str(order)], data.bragg_info2['bragg_inten_err_order'+str(order)], label=info)
        plt.ylabel(rf'$\Delta I / I_0$')
        plt.xlabel('Delay (ps)')


class scan:
    """
    Inputs: path (str): directory with relevant bragginfo dicts, order_skip txt file, t0 txt file, etc. 
    Needs: numpy as np, pandas as pd, scipy.optimze.curve_fit as curve_fit, material, Q
    Loads data, computes rms, fits log for t = delay[21]
    """
    def __init__(self, path):
        """
        
        """
        self.data1 = {}
        self.data2 = {}
        if 'order_skip.txt' in os.listdir(path) and os.path.getsize(os.path.join(path, 'order_skip.txt')) != 0:
            self.order_skip = [int(i) for i in pd.read_csv(os.path.join(path, 'order_skip.txt')).columns]
        else:
            self.order_skip = []
            
        if 'data_description.txt' in os.listdir(path):
            descr  = pd.read_csv(os.path.join(path, 'data_description.txt'), delimiter='_')
            if 'ML' in descr.columns:
                self.type, self.pump, self.fluence, self.temp, self.t0 = descr.columns[1:]
                self.deg = ''
                for file in os.listdir(path):
                    scan_num = os.path.basename(path)
                    
                    if file.startswith(scan_num + '_bragginfo1'):
                        bragg_info1 = pd.read_csv(os.path.join(path,file))
                        mat = material(os.path.join(path,file))
                for i in np.arange(9,-1,-1):
                    bragg_info1['log_inten_order'+str(i+1)] = np.log(bragg_info1['bragg_inten_mean_order'+str(i+1)])
                    bragg_info1['log_inten_err_order'+str(i+1)] = bragg_info1['bragg_inten_err_order'+str(i+1)] / bragg_info1['bragg_inten_mean_order'+str(i+1)]
                self.bragg_info1 = bragg_info1
                self.bragg_info1.mat = mat
                self.delay = pd.read_csv(os.path.join(path, 'delay_time_t0.txt'))['delay_time_t0']
                    
            else:
                self.deg, self.pump, self.fluence, self.temp, self.t0 = descr.columns[2:]
                self.deg = float(self.deg.replace('deg', ''))
                self.type = 'HS'

                for file in os.listdir(path):
                    scan_num = os.path.basename(path)
                    if file.startswith(scan_num + '_bragginfo1'):
                        bragg_info1 = pd.read_csv(os.path.join(path,file))
                        mat1 = material(os.path.join(path,file))
                    if file.startswith(scan_num + '_bragginfo2'):
                        bragg_info2 = pd.read_csv(os.path.join(path,file))
                        mat2 = material(os.path.join(path,file))
                for i in np.arange(9,-1,-1):
                    bragg_info1['log_inten_order'+str(i+1)] = np.log(bragg_info1['bragg_inten_mean_order'+str(i+1)])
                    bragg_info1['log_inten_err_order'+str(i+1)] = bragg_info1['bragg_inten_err_order'+str(i+1)] / bragg_info1['bragg_inten_mean_order'+str(i+1)]
                    bragg_info2['log_inten_order'+str(i+1)] = np.log(bragg_info2['bragg_inten_mean_order'+str(i+1)])
                    bragg_info2['log_inten_err_order'+str(i+1)] = bragg_info2['bragg_inten_err_order'+str(i+1)] / bragg_info2['bragg_inten_mean_order'+str(i+1)]
                    
                self.bragg_info1 = bragg_info1
                self.bragg_info2 = bragg_info2 
                self.bragg_info1.mat = mat1
                self.bragg_info2.mat = mat2
                self.delay = pd.read_csv(os.path.join(path, 'delay_time_t0.txt'))['delay_time_t0']


        else:
            print('data_description not found')

    def q_arr(self, order_num):
        a_dict = {'MoSe2': 3.32, 'WSe2': 3.32, 'MoS2':3.18, 'WS2':3.18}
        bo_dict = {'order1' : [1,0,0], 'order2' : [2,-1,0], 
           'order3' : [2,0,0], 'order4' :[3,-1,0], 
           'order5' : [3,0,0], 'order6' : [4,-2,0], 
           'order7' : [4,-1,0], 'order8' : [4,0,0], 
           'order9' : [5,-2,0], 'order10' : [5,-1,0]}
        
        order_arr = np.arange(order_num + 1)
        order_arr = np.delete(order_arr, self.order_skip)[1:]
        self.order_arr = order_arr
        
        self.q1_arr = np.array([Q(i, a_dict[self.bragg_info1.mat]) for i in order_arr])
        self.data1['q_sq'] = self.q1_arr**2
        
        if self.type != 'ML':
            self.q2_arr = np.array([Q(i, a_dict[self.bragg_info2.mat]) for i in order_arr])
            self.data2['q_sq'] = self.q2_arr**2 
        
    def fit_log(self, idx=21, plot=False, order_num=10):
        """
        Obtain -log(I_t/I_0) and fit vs q^2
        inputs: self, order_num (int, typically 10), order_skip (list of int, orders that may have overlapping peaks to skip)
                time (int) delay time value to compute the ratio at 
        outputs: u_rms, 
        """
        self.q_arr(order_num)
        
        log_inten_1 = np.array([])
        log_inten_1_err = np.array([])
        for i in self.order_arr:
            log_inten_1 = np.append(log_inten_1, (self.bragg_info1['log_inten_order'+str(i)][idx]))
            log_inten_1_err = np.append(log_inten_1_err, (self.bragg_info1['log_inten_err_order'+str(i)][idx]))
        
        x1 = self.data1['q_sq']
        y1 = -1*(log_inten_1)
        
        popt1, pcov1 = curve_fit(linear, x1, y1, sigma=log_inten_1_err, absolute_sigma=True)
        perr1 = np.sqrt(np.diag(pcov1))

        nstd = 2
        popt1_up = popt1 + nstd * perr1
        popt1_dw = popt1 - nstd * perr1
        fit1_up = linear(x1, *popt1_up)
        fit1_dw = linear(x1, *popt1_dw)
        
        self.data1['log_inten_1'] = log_inten_1
        self.data1['log_inten_1_err'] = log_inten_1_err
        self.data1['log_fit1'] = linear(x1, *popt1)
        self.data1['log_fit1_up'] = fit1_up
        self.data1['log_fit1_dw'] = fit1_dw
        self.data1['params'] = popt1
        
        if self.type != 'ML':
            
            log_inten_2 = np.array([])
            log_inten_2_err = np.array([])
            for i in self.order_arr:
                log_inten_2 = np.append(log_inten_2, (self.bragg_info2['log_inten_order'+str(i)][idx]))
                log_inten_2_err = np.append(log_inten_2_err, (self.bragg_info2['log_inten_err_order'+str(i)][idx]))
            
            x2 = self.data2['q_sq']
            y2 = -1*(log_inten_2)

            popt2, pcov2 = curve_fit(linear, x2, y2, sigma=log_inten_2_err, absolute_sigma=True)
            perr2 = np.sqrt(np.diag(pcov2))

            nstd = 2
            popt2_up = popt2 + nstd * perr2
            popt2_dw = popt2 - nstd * perr2
            fit2_up = linear(x2, *popt2_up)
            fit2_dw = linear(x2, *popt2_dw)
            
            self.data2['log_inten_2'] = log_inten_2 
            self.data2['log_inten_2_err'] = log_inten_2_err
            self.data2['log_fit2'] = linear(x2, *popt2)
            self.data2['log_fit2_up'] = fit2_up
            self.data2['log_fit2_dw'] = fit2_dw
            self.data2['params'] = popt2   
        if plot: 
            if self.type != 'ML':
                plt.scatter(self.data1['q_sq'], -1*self.data1['log_inten_1'], color='k', label=self.bragg_info1.mat+'_'+str(self.deg)+'_'+self.temp+'_'+self.fluence) 
                plt.errorbar(self.data1['q_sq'], -1*self.data1['log_inten_1'], self.data1['log_inten_1_err'], color='k', linestyle='')
                plt.plot(self.data1['q_sq'], self.data1['log_fit1'], 'k') 
                plt.scatter(self.data2['q_sq'], -1*self.data2['log_inten_2'], color='r', label=self.bragg_info2.mat+'_'+str(self.deg)+'_'+self.temp+'_'+self.fluence) 
                plt.errorbar(self.data2['q_sq'], -1*self.data2['log_inten_2'], self.data2['log_inten_2_err'], color='r', linestyle='')
                plt.plot(self.data2['q_sq'], self.data2['log_fit2'], 'r')
            else: 
                plt.scatter(self.data1['q_sq'], -1*self.data1['log_inten_1'], color='k', label=self.bragg_info1.mat+'_ML_'+self.temp+'_'+self.fluence) 
                plt.errorbar(self.data1['q_sq'], -1*self.data1['log_inten_1'], self.data1['log_inten_1_err'], color='k', linestyle='')
                plt.plot(self.data1['q_sq'], self.data1['log_fit1'], color='k')
            plt.legend(fontsize=15)
            plt.xlabel(r'$Q^2 [Å^{-2}]$', size=20) 
            plt.ylabel(r'$-ln(I/I_0)$', size=20)
            plt.xticks(size=20);
            plt.yticks(size=20);
                       
                    
    def rms(self, order_num=10):
        """
        Compute the log intensity change and find dependence on Q^2 fitting a line to obtain u_rms; do this for 
        every time t and obtain u vs. t data. 
        Outputs: self.rms1, self.rms1_err, self.rms2, self.rms2_err (if heterobilayer) 
        """
        self.q_arr(order_num)
        rms1 = []
        for i, t in enumerate(self.delay):
            log_inten_1 = []
            log_inten_1_err = []
            for j in self.order_arr:
                log_inten_1.append(self.bragg_info1['log_inten_order'+str(j)][i])
                log_inten_1_err.append(self.bragg_info1['log_inten_err_order'+str(j)][i])
            
            log_inten_1_err = np.array(log_inten_1_err)
            log_inten_1 = np.array(log_inten_1)
            popt1, pcov1 = curve_fit(linear, 0.5*self.data1['q_sq'], -1*log_inten_1, sigma=log_inten_1_err, absolute_sigma=True)
            perr1 = np.sqrt(np.diag(pcov1))
            rms1.append((popt1[0], perr1[0]))

        rms1 = np.array(rms1)
        self.rms1 = rms1[:,0]
        self.rms1_err = rms1[:,1]

        if self.type != 'ML':
            rms2 = []
            for i, t in enumerate(self.delay):
                log_inten_2 = []
                log_inten_2_err = []
                for j in self.order_arr:
                    log_inten_2.append(self.bragg_info2['log_inten_order'+str(j)][i])
                    log_inten_2_err.append(self.bragg_info2['log_inten_err_order'+str(j)][i])
                
                log_inten_2_err = np.array(log_inten_2_err)
                log_inten_2 = np.array(log_inten_2)
                popt2, pcov2 = curve_fit(linear, 0.5*self.data2['q_sq'], -1*log_inten_2, sigma=log_inten_2_err, absolute_sigma=True)
                perr2 = np.sqrt(np.diag(pcov2))
                rms2.append((popt2[0], perr2[0]))

            rms2 = np.array(rms2)
            self.rms2 = rms2[:,0]
            self.rms2_err = rms2[:,1]


def plot_pretty_individual(names, m, w, comparison, ax1, ax2, fit_type_m = 'rd', fit_type_w='rrd', norm=False, Bin=False, ML_names=[], error=False, plot_fit=False, inset=False, idx=(5, 1)):
    """
    Plot t vs rms for a set of scans with m and w on separate plots to compare same dynamics of same material in different layers
    Takes axes for flexibility with where and how the plot looks
    Inputs: 
        names(list): scan numbers to plot e.g 20211010_0006, if plotting just ML, names= []
        m(str): 'MoSe2' or 'MoS2'
        w(str): 'WSe2' or 'WS2'
        comparison: 'deg', 'pump', 'fluence', 'temp' legend parameters to show what is different between scans 
        ax1 (matplotlib.axes._subplots.AxesSubplot): subplot for mo scans
        ax2 (matplotlib.axes._subplots.AxesSubplot): subplot for w scans
        fit_type_w: 'rd', 'rrd' (default) what type of fit to plut for W layer (rise_decay, rise_rise_decay)
        fit_type_m: 'rd' (default), 'rdd' what type of fit to plut for Mo layer (rise_decay, rise_decay_decay)
        norm(bool): scale all data to have maximum 1; default:False
        Bin(bool): plot binned data; default: False
        ML_names(list): list of ML scan numbers to additionally plot; default: empty list
        error(bool): plot error; default:False
        plot_fit(bool): plot best fit line from default params in directory; default:False
        inset(bool): add inset showing range of longtimes; default:False
        idx(tup): range of what values around maximum to average to 1; default:(max-5:max+1)
    
    Outputs: 
        t vs. rms for m and w with m and w on separate plots 
    """
    colors = []
    idx1, idx2 = idx
    cmap = matplotlib.cm.get_cmap('viridis', 5)
    for i in range(cmap.N):
        rgba = cmap(i)
        colors.append(matplotlib.colors.rgb2hex(rgba))
    if plot_fit: 
        colors = [c for c in colors for _ in (0, 1)]
        new_cycler = (cycler(color=colors))
    else: 
        new_cycler = (cycler(color=colors))
    plt.rc('axes', prop_cycle=new_cycler)

        
    if inset: 
        axins1 = inset_axes(ax1, width='30%', height='30%', loc=4,
                   bbox_to_anchor=(-0.05,0.11,1, 1), bbox_transform=ax1.transAxes)
        axins2 = inset_axes(ax2, width='30%', height='30%', loc=4, 
                   bbox_to_anchor=(-0.05,0.11,1, 1), bbox_transform=ax2.transAxes)
        
        axins1.set_xlabel('(ps)', size=12)
        axins1.set_ylabel(r'Norm $\Delta \langle u_{ip}^2 \rangle [\AA^2]$', size=12)
        axins2.set_xlabel('(ps)', size=12)
        axins2.set_ylabel(r'Norm $\Delta \langle u_{ip}^2 \rangle [\AA^2]$', size=12)
    for name in names:
        data = scan(path + name)
        data.rms()
        compare_dict = {'deg':str(data.deg) + ' deg', 'pump':data.pump, 'fluence':data.fluence, 'temp':data.temp}
        label = compare_dict[comparison[0]]
        if len(comparison) > 1: 
            i = 1
            while i < len(comparison):
                label += ' ' + compare_dict[comparison[i]]
                i+=1
        if name in t0_dict.keys():
            t0 = t0_dict[name]
        else: 
            t0 = input()
        if not Bin: 
            delay = data.delay - t0
        if data.bragg_info1.mat == m:
            if norm: 
                idx = np.argmax(data.rms1)
                vals = data.rms1[idx-idx1:idx+idx2]
                fac = np.nanmean(vals) 
            else:
                fac = 1
            if Bin: 
                temp = fit(path + name, plot=False)
                temp.Bin(1, 20, bin_limit='tot',  plot=False)
                delay = temp.binned_t
                data.rms1 = temp.binned_vals
                data.rms1_err = temp.binned_errs
            if error: 
                ax1.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', label=label)
                ax1.fill_between(delay, data.rms1 * 1/fac + data.rms1_err * 1/fac, data.rms1 * 1/fac - data.rms1_err * 1/fac, alpha=0.5)
            if plot_fit: 
                mask = delay >= 0 
                if fit_type_m == 'rd': 
                    params, err = fit_params_rd(name, 1) 
                    y = rise_decay(delay, *params) * 1/fac
                    y[y < 0] = 0
                    ax1.plot(delay - t0, y)
                    if inset: 
                        axins1.plot(delay - t0, y)
                    
                elif fit_type_m == 'rdd': 
                    params, err = fit_params_rdd(name, 1)  
                    y = rise_decay_decay(delay, *params) * 1/fac
                    y[y< 0] = 0 
                    ax1.plot(delay - t0, y)    
                    if inset: 
                        axins1.plot(delay - t0, y)    
            if not error or plot_fit: 
                ax1.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', label=label)
                if inset: 
                    axins1.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, markersize=2, linestyle='', label=label)

        if data.bragg_info2.mat == m:
            if norm: 
                idx = np.argmax(data.rms2)
                vals = data.rms2[idx-idx1:idx+idx2]
                fac = np.nanmean(vals) 
            else:
                fac = 1
            if Bin: 
                temp = fit(path + name, plot=False)
                temp.Bin(2, 20, bin_limit='tot',  plot=False)
                delay = temp.binned_t
                data.rms2 = temp.binned_vals
                data.rms2_err = temp.binned_errs
            if error: 
                ax1.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', label=label)
                ax1.fill_between(delay, data.rms2 * 1/fac + data.rms2_err * 1/fac, data.rms2 * 1/fac - data.rms2_err * 1/fac, alpha=0.5)
            if plot_fit: 
                mask = delay >= 0 
                if fit_type_m == 'rd': 
                    params, err = fit_params_rd(name, 2) 
                    y = rise_decay(delay, *params) * 1/fac
                    y[y < 0] = 0 
                    ax1.plot(delay - t0, y)
                    if inset: 
                        axins1.plot(delay - t0, y)
                elif fit_type_m == 'rdd': 
                    params, err = fit_params_rdd(name, 2) 
                    y = rise_rise_decay(delay, *params) * 1/fac
                    y[y < 0] = 0
                    ax1.plot(delay - t0, y)   
                    if inset: 
                        axins1.plot(delay - t0, y)
            if not error or plot_fit:
                ax1.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', label=label)
                if inset: 
                    axins1.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, markersize=2, linestyle='', label=label)

        if data.bragg_info1.mat == w:
            if norm: 
                idx = np.argmax(data.rms1)
                vals = data.rms1[idx-idx1:idx+idx2]
                fac = np.nanmean(vals) 
            else:
                fac = 1
            if Bin: 
                temp = fit(path + name, plot=False)
                temp.Bin(1, 20, bin_limit='tot',  plot=False)
                delay = temp.binned_t
                data.rms1 = temp.binned_vals
                data.rms1_err = temp.binned_errs
            if error: 
                ax2.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', label=label)
                ax2.fill_between(delay, data.rms1 * 1/fac + data.rms1_err * 1/fac, data.rms1 * 1/fac - data.rms1_err * 1/fac, alpha=0.5)
            if plot_fit: 
                mask = delay >= 0 
                
                if fit_type_w == 'rd':
                    params, err = fit_params_rd(name, 1) 
                    y = rise_decay(delay, *params) * 1/fac
                    y[y < 0] = 0 
                    ax2.plot(delay - t0, y)
                    if inset: 
                        axins2.plot(delay - t0, y)
                elif fit_type_w == 'rrd': 
                    params, err = fit_params_rrd(name, 1) 
                    y = rise_rise_decay(delay, *params) * 1/fac
                    y[y < 0] = 0
                    ax2.plot(delay - t0, y)   
                    if inset: 
                        axins2.plot(delay - t0, y)
            if not error or plot_fit:
                ax2.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', label=label)
                if inset: 
                    axins2.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, markersize=2, linestyle='', label=label)

        if data.bragg_info2.mat == w:
            if norm: 
                idx = np.argmax(data.rms2[1:])
                vals = data.rms2[idx-idx1:idx+idx2]
                fac = np.nanmean(vals) 
            else:
                fac = 1
            if Bin: 
                temp = fit(path + name, plot=False)
                temp.Bin(2,20, bin_limit='tot',  plot=False)
                delay = temp.binned_t
                data.rms2 = temp.binned_vals
                data.rms2_err = temp.binned_errs
            if error: 
                ax2.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', label=label)
                ax2.fill_between(delay, data.rms2 * 1/fac + data.rms2_err * 1/fac, data.rms2 * 1/fac - data.rms2_err * 1/fac, alpha=0.5)
            if plot_fit: 
                mask = delay >= 0 
                if fit_type_w == 'rd':
                    params, err = fit_params_rd(name, 2)
                    y = rise_decay(delay, *params) * 1/fac
                    y[y < 0] = 0 
                    ax2.plot(delay - t0, y)
                    if inset: 
                        axins2.plot(delay - t0, y)
                elif fit_type_w == 'rrd': 
                    params, err = fit_params_rrd(name, 2)
                    y = rise_rise_decay(delay, *params) * 1/fac
                    y[y < 0 ] = 0
                    ax2.plot(delay - t0, y)  
                    if inset:  
                        axins2.plot(delay - t0, y)
            if not error or plot_fit:
                ax2.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', label=label)
                if inset: 
                    axins2.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, markersize=2, linestyle='', label=label)
    if len(ML_names) > 0:     
        for name in ML_names:
            data = scan(path + name)
            data.rms()
            if name in t0_dict.keys():
                t0 = t0_dict[name]
            else: 
                t0 = input()
            if not Bin: 
                delay = data.delay - t0
            if data.bragg_info1.mat == m:
                if norm: 
                    idx = np.argmax(data.rms1)
                    vals = data.rms1[idx-idx1:idx+idx2]
                    fac = np.nanmean(vals) 
                else:
                    fac = 1
                if Bin: 
                    temp = fit(path + name, plot=False)
                    temp.Bin(1, 30, bin_limit='tot',  plot=False)
                    delay = temp.binned_t
                    data.rms1 = temp.binned_vals
                    data.rms1_err = temp.binned_errs
                if error: 
                    ax1.plot(delay, data.rms1 * 1/fac, label='Monolayer', color='k', marker='o', mew=2, markersize=4, linestyle = '')
                    ax1.fill_between(delay, data.rms1 * 1/fac + data.rms1_err * 1/fac, data.rms1 * 1/fac - data.rms1_err * 1/fac, color='k', alpha=0.5)
                if plot_fit: 
                    mask = delay >= 0 
                    params, err = fit_params_rd(name, 1) 
                    if len(params) <=5 : 
                        y = rise_decay(delay, *params) * 1/fac
                        y[y < 0] = 0 
                        ax1.plot(delay - t0, y, color = 'k')
                        if inset: 
                            axins1.plot(delay - t0, y, color='k')
                    else: 
                        y = rise_rise_decay(delay, *params) * 1/fac
                        y[y < 0] = 0 
                        ax1.plot(delay - t0, y, color = 'k')  
                        if inset:  
                            axins1.plot(delay - t0, y, color = 'k')
                if not error or plot_fit:
                    ax1.plot(delay, data.rms1 * 1/fac, label='Monolayer', color='k', marker='o', mew=2, markersize=4, linestyle = '')
                    if inset: 
                        axins1.plot(delay, data.rms1 * 1/fac, label='Monolayer', color='k', marker='o', mew=2, markersize=2, linestyle = '')

            if data.bragg_info1.mat == w: 
                if w == 'WS2':
                    data.rms()
                    data.delay = data.delay[:-1]
                    data.rms1 = data.rms1[:-1] - np.nanmean(data.rms1[-8:-1])
                    data.rms1_err = data.rms1_err[:-1]
                    delay = data.delay - t0
                if norm: 
                    idx = np.argmax(data.rms1)
                    vals = data.rms1[idx-idx1:idx+idx2]
                    fac = np.nanmean(vals) 
                else:
                    fac = 1
                if Bin: 
                    temp = fit(path + name, plot=False)
                    temp.Bin(1, 20, bin_limit='tot',  plot=False)
                    delay = temp.binned_t
                    data.rms1 = temp.binned_vals
                    data.rms1_err = temp.binned_errs
                if error: 
                    ax2.plot(delay, data.rms1 * 1/fac, label='Monolayer', color='k', marker='o', mew=2, markersize=4, linestyle = '')
                    ax2.fill_between(delay, data.rms1 * 1/fac + data.rms1_err * 1/fac, data.rms1 * 1/fac - data.rms1_err * 1/fac, color='k', alpha=0.3)
                if plot_fit: 
                    mask = delay >= 0 
                    params, err = fit_params_rd(name, 1) 
                    if len(params) <=5 : 
                        y = rise_decay(delay, *params) * 1/fac
                        y[y < 0] = 0 
                        ax2.plot(delay - t0, y, color='k')
                        if inset: 
                            axins2.plot(delay - t0, y, color='k')
                    else: 
                        y = rise_rise_decay(delay, *params) * 1/fac
                        y[y < 0] = 0 
                        ax2.plot(delay - t0, y, color='k')   
                        if inset: 
                            axins2.plot(delay - t0,  y, color='k')
                if not error or plot_fit:
                    ax2.plot(delay, data.rms1 * 1/fac, label='Monolayer', color='k', marker='o', mew=2, markersize=4, linestyle = '')
                    if inset: 
                        axins2.plot(delay, data.rms1 * 1/fac, label='Monolayer', color='k', marker='o', mew=2, markersize=2, linestyle = '')


