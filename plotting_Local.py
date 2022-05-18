#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from UED_Analysis_Functions_Local import *
import matplotlib_specs
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib 
import copy
from fitting_script import fit
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

path = '/home/amalyajohnson/Research/UED_U085/Processed_Data/'

def rise_rise_decay(x, y0, A, a, x0, tau_1, tau_3, tau_2): 
    """
    Exponential double rise & decay convolution with sigma = 0.15; instrument resolution 
    Inputs: 
            x: independent paramater for fit (time in rms atomic displacement data)
            y0: y offset
            A: amplitude 
            a: contributition of tau_3 to fit [0,1]
            x0: x offset 
            tau_1: fast rise compononent
            tau_3: slow rise component
            tau_2: slow decay component 
    Outputs: exponential double rise & decay curve
    """
    return y0 + A*(-(1-a)*np.exp(-(x-x0)/tau_1 + (0.15**2)/(2*tau_1**2)) + -a*np.exp(-(x-x0)/tau_3 + (0.15**2)/(2*tau_3**2)) + np.exp(-(x-x0)/tau_2 + (0.15**2)/(2*tau_2**2)))

def rise_decay_decay(x, y0, A, a, x0, tau_1, tau_3, tau_2): 
    """
    Exponential rise & double decay convolution with sigma = 0.15; instrument resolution 
    Inputs: 
            x: independent paramater for fit (time in rms atomic displacement data)
            y0: y offset
            A: amplitude 
            a: contribution of tau_3 to fit [0,1]
            x0: x offset 
            tau_1: fast rise compononent
            tau_3: slow decay component
            tau_2: slower decay component 
    Outputs: exponential rise & double decay curve
    """
    return y0 + A*(-np.exp(-(x-x0)/tau_1 + (0.15**2)/(2*tau_1**2)) + a*np.exp(-(x-x0)/tau_3 + (0.15**2)/(2*tau_3**2)) + (1-a)*np.exp(-(x-x0)/tau_2 + (0.15**2)/(2*tau_2**2)))

def fit_params_rdd(name, info): 
    global path
    fn = path + name
    data = scan(fn)
    if info ==1:
        params = pd.read_csv(f'{fn}/bragginfo1_fitparams_rdd.txt')['val']
        err = pd.read_csv(f'{fn}/bragginfo1_fitparams_rdd.txt')['err'] 
    if info ==2: 
        params = pd.read_csv(f'{fn}/bragginfo2_fitparams_rdd.txt')['val']
        err = pd.read_csv(f'{fn}/bragginfo2_fitparams_rdd.txt')['err']
    return params, err 

def fit_params_ht3(name, info): 
    global path
    fn = path + name
    data = scan(fn)
    if info ==1:
        params = pd.read_csv(f'{fn}/bragginfo1_fitparams_ht3.txt')['val']
        err = pd.read_csv(f'{fn}/bragginfo1_fitparams_ht3.txt')['err'] 
    if info ==2: 
        params = pd.read_csv(f'{fn}/bragginfo2_fitparams_ht3.txt')['val']
        err = pd.read_csv(f'{fn}/bragginfo2_fitparams_ht3.txt')['err']
    return params, err 


def fit_params(name, info): 
    global path
    fn = path + name
    data = scan(fn)
    if info ==1:
        params = pd.read_csv(f'{fn}/bragginfo1_fitparams.txt')['val']
        err = pd.read_csv(f'{fn}/bragginfo1_fitparams.txt')['err']
    if info ==2: 
        params = pd.read_csv(f'{fn}/bragginfo2_fitparams.txt')['val']
        err = pd.read_csv(f'{fn}/bragginfo2_fitparams.txt')['err']
    return params, err 

def fit_params_rrd(name, info): 
    global path
    fn = path + name
    data = scan(fn)
    if info ==1:
        params = pd.read_csv(f'{fn}/bragginfo1_fitparams_rrd.txt')['val']
        err = pd.read_csv(f'{fn}/bragginfo1_fitparams_rrd.txt')['err']
    if info ==2: 
        params = pd.read_csv(f'{fn}/bragginfo2_fitparams_rrd.txt')['val']
        err = pd.read_csv(f'{fn}/bragginfo2_fitparams_rrd.txt')['err']
    return params, err 

def fit_params_rd(name, info): 
    global path
    fn = path + name
    data = scan(fn)
    if info ==1:
        params = pd.read_csv(f'{fn}/bragginfo1_fitparams_rd.txt')['val']
        err = pd.read_csv(f'{fn}/bragginfo1_fitparams_rd.txt')['err']
    if info ==2: 
        params = pd.read_csv(f'{fn}/bragginfo2_fitparams_rd.txt')['val']
        err = pd.read_csv(f'{fn}/bragginfo2_fitparams_rd.txt')['err']
    return params, err 

t0_dict = {'20211010_1021':-0.1, '20211009_2204':0.1, 
          '20211010_0006': 0.1, '20211009_1623':0.0,
          '20211008_0116':-0.1, '20211007_1521':-0.2, 
          '20211009_2129':0.2, '20211009_2101':0.2, 
          '20211009_1848':0.0, '20211008_1603':0.0, 
          '20211008_1436':0.0, '20211008_1423':0.0, 
          '20211007_1416':0.6, '20211007_1314':0.0, 
          '20211009_1124':0.3, '20211009_0337':0.0, 
          '20211009_0011':0.1, '20211008_2113':0.1, 
          '20211008_1906':0.1, '20211007_2246': 0.0, 
          '20211007_1753':-0.3, '20211007_0924':-0.2, 
          '20211007_0258':-0.2, '20211007_0102':-0.1, 
          '20211006_2235':0.0, '20211006_2012':0.0,
          '20211006_1809':-0.2, '20211006_1615':-0.1,
          '20211010_0903':0.1, '20211010_1107':-0.1,
          '20211010_1021':0.0, '20211008_0116':0.0,
          '20211006_1328':-0.1, '20211006_1246':-0.2,
          '20211005_2304':-0.2, '20211008_1755':0.0, 
          '20211008_0939':0.0, '20211006_0731':0.0, 
          '20211005_1759':-0.2}

def plot_pretty(names, m, w, comparison, fit_type_m = 'rd', fit_type_w='rrd', shortM = False, shortW=False, norm=False, Bin=False, ML_names=[], error=False,
title = str(), plot_fit=False, legend=True, inset=False, idx=(5, 1)):
    """
    Plot t vs rms for a set of scans with m and w on separate plots to compare same dynamics of same material in different layers

    Inputs: 
        names(list): scan numbers to plot e.g 20211010_0006
        m(str): 'MoSe2' or 'MoS2'
        w(str): 'WSe2' or 'WS2'
        comparison: 'deg', 'pump', 'fluence', 'temp' legend parameters to show what is different between scans 
        fit_type_w: 'rd', 'rrd' (default) what type of fit to plut for W layer (rise_decay, rise_rise_decay)
        fit_type_m: 'rd' (default), 'rdd' what type of fit to plut for Mo layer (rise_decay, rise_decay_decay)
        shortM(tup): x range for m plot;  default: False, plot full range
        shortW(tup): x range for w plot; default: False, plot full range 
        norm(bool): scale all data to have maximum 1; default:False
        Bin(bool): plot binned data; default: False
        ML_names(list): list of ML scan numbers to additionally plot; default: empty list
        error(bool): plot error; default:False
        title(str): title of plot; default: empty string
        plot_fit(bool): plot best fit line from default params in directory; default:False
        legend(bool): show legend; default: True
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
#     colors = colors[::-1]    
    if len(names) == 2: 
        colors = ['k', 'r']
    if plot_fit: 
        colors = [c for c in colors for _ in (0, 1)]
        new_cycler = (cycler(color=colors))
    else: 
        new_cycler = (cycler(color=colors))
    plt.rc('axes', prop_cycle=new_cycler)
    mtitle = m.replace('2', '$_2$') + title 
    wtitle = w.replace('2', '$_2$') + title

    if legend and inset: 
        f, (ax1, ax2) = plt.subplots(2,1, sharey=False, figsize=(7,10))
    else: 
        f, (ax1, ax2) = plt.subplots(2, 1, sharey=False, figsize=(5,10))
        
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

    if legend and inset: 
        ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        ax2.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    elif legend: 
        ax1.legend()
        ax2.legend()
    if shortM !=False:
        ax1.set_xlim(shortM)
    if shortW !=False:
        ax2.set_xlim(shortW)
    
    if norm: 
        ylabel = r'Normalized $\Delta \langle u_{ip}^2 \rangle [\AA^2]$'
    else: 
        ylabel = r'$\Delta \langle u_{ip}^2 \rangle [\AA^2]$'
    ax1.set_ylabel(ylabel , fontsize=15)
    ax2.set_ylabel(ylabel, fontsize=15)



    ax1.set_xlabel('Delay Time (ps)')
    ax1.set_xlabel('Delay Time (ps)')
    ax2.set_xlabel('Delay Time (ps)')
    ax1.set_title(mtitle)
    ax2.set_title(wtitle)
    plt.tight_layout()

def plot_tau_single(mat, compare, names, fit_type, ML_names = []):
    """ 
    Plot all tau (tau_1 and tau_2, tau_3 and a if applicable) for a specific material in a set of scans

    Inputs: 
        mat(str): material; 'MoS2', 'MoSe2', 'WS2', 'WSe2'
        compare(str): value for x axis 'deg', 'temp', 'power'
        names(list): scan numbers to plot e.g 20211010_0006
        fit_type: 'rd' (rise_decay), 'rdd' (rise_decay_decay for Mo), 'rrd' (rise_rise_decay for W) which fitting model tau to plot
        ML_names(list): list of ML scan numbers to additionally plot; default: empty list
    
    Outputs: 
        set of tau vs. compare value for specific material
    
    """
    labels = {'y0':0, 'A':1, 'a':2, 'x0':3, 'tau_1':4, 'tau_2':5, 'tau_3':6}
    labels_rd = {'y0':0, 'A':1, 'a':None, 'x0':2, 'tau_1':3, 'tau_2':None, 'tau_3':4}
    ylabel_dict = {'a':r'$\tau_2$ contribution', 'tau_1':r'$\tau_1$ [ps]', 'tau_2':r'$\tau_2$ [ps]', 'tau_3':r'$\tau_3$ [ps]'}

    colors = []
    cmap = matplotlib.cm.get_cmap('viridis', 5)
    for i in range(cmap.N):
        rgba = cmap(i)
        colors.append(matplotlib.colors.rgb2hex(rgba))
    if len(names) == 2: 
        colors = ['k', 'r']
    new_cycler = (cycler(color=colors))
    plt.rc('axes', prop_cycle=new_cycler)


    if fit_type == 'rd': 
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(12,6))
        for val, ax in zip(['tau_1', 'tau_3'], [ax1, ax2]):
            xrange = []
            for name in names:
                idx = labels_rd[val]
                # print('using rise decay for Mo')
                data = scan(path + name)
                if compare == 'power': 
                    x = float(data.fluence.replace('mj', ''))
                    xrange.append(x)
                    ax.set_xlabel(rf'Fluence [mj cm$^{-2}$ ]')
                if compare == 'deg': 
                    x = data.deg
                    xrange.append(x)
                    ax.set_xlabel(rf'Twist Angle [$\Delta \phi$]')
                if compare == 'temp':
                    x = int(data.temp.replace('K', ''))
                    xrange.append(x)
                    ax.set_xlabel(rf'Temperature [K]')
                if data.bragg_info1.mat == mat:
                    params, err = fit_params_rd(name, 1)
                    y = params[idx]
                    err = err[idx]
                    ax.scatter(x, y)
                    ax.errorbar(x, y, err)
                if data.bragg_info2.mat == mat:
                    params, err = fit_params_rd(name, 2)
                    y = params[idx]
                    err = err[idx]
                    ax.scatter(x, y)
                    ax.errorbar(x, y, err)
            points = np.arange(min(xrange) - 1, max(xrange) + 3)
            for name in ML_names: 
                data = scan(path + name)
                idx = labels_rd[val]
                if data.bragg_info1.mat == mat:
                    params, err = fit_params_rd(name, 1)
                    err = err[idx]
                    y = params[idx]
                    ax.fill_between(points, np.repeat(y - err, len(points)), np.repeat(y + err, len(points)), color='k', alpha=0.3)
                    ax.hlines(y, min(xrange) - 1, max(xrange) + 2, color = 'k', linestyles='--')
            title = mat.replace('2', '_2')
            ax.set_title(mat)
            ax.set_ylabel(ylabel_dict[val])
        plt.tight_layout()


    else: 
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, sharey=False, figsize=(20,5))
        for val, ax in zip(['a', 'tau_1', 'tau_2', 'tau_3'], [ax1, ax2, ax3, ax4]):
            xrange = []
            for name in names: 
                data = scan(path + name)
                data.rms()
                if compare == 'power': 
                    x = float(data.fluence.replace('mj', ''))
                    xrange.append(x)
                    ax.set_xlabel(rf'power [$mj /cm^{2}$ ]')
                if compare == 'deg': 
                    x = data.deg
                    xrange.append(x)
                    ax.set_xlabel(rf'Twist Angle [$\Delta \phi$]')
                if compare == 'temp':
                    x = int(data.temp.replace('K', ''))
                    xrange.append(x)
                    ax.set_xlabel(rf'Temperature [K]')
                if data.bragg_info1.mat == mat:
                    if 'Mo' in mat: 
                        params, err = fit_params_rdd(name, 1)
                        idx = labels[val]
                    if 'W' in mat: 
                        params, err = fit_params_rrd(name, 1)
                        idx = labels[val]
                    if idx != None: 
                        y = params[idx]
                        err = err[idx]
                        ax.scatter(x, y)
                        ax.errorbar(x, y, err)
                if data.bragg_info2.mat == mat:
                    if 'Mo' in mat: 
                        params, err = fit_params_rdd(name, 2)
                        idx = labels[val]
                        # if idx == None: 
                        #     pass
                    if 'W' in mat: 
                        params, err = fit_params_rrd(name, 2)
                        idx = labels[val]
                        # if idx == None: 
                        #     pass 
                    if idx != None: 
                        y = params[idx]
                        err = err[idx]
                        ax.scatter(x, y)
                        ax.errorbar(x, y, err)
            points = np.arange(min(xrange) - 1, max(xrange) + 3)
            for name in ML_names: 
                data = scan(path + name)
                idx = labels_rd[val]
                if data.bragg_info1.mat == mat and idx != None:
                    params, err = fit_params_rd(name, 1)
                    err = err[idx]
                    y = params[idx]
                    ax.fill_between(points, np.repeat(y - err, len(points)), np.repeat(y + err, len(points)), color='k', alpha=0.3)
                    ax.hlines(y, min(xrange) - 1, max(xrange) + 2, color = 'k', linestyles='--')
            title = mat.replace('2', '$_2$')
            ax.set_title(title)
            ax.set_ylabel(ylabel_dict[val])
        plt.tight_layout()
        

def plot_m_w(names, m, w, comparison, fit_type_m='rd', fit_type_w='rrd', norm=False, Bin=True, shorttime=False, ML=[], title=str(), error=True, plot_fit = True, legend=True, 
inset = False):
    """
    Plot t vs. rms for m and w on same plot to compare how dynamics m and w differ within the same heterostructure 

    Inputs: 
        names(list): scan numbers to plot e.g 20211010_0006
        m(str): 'MoSe2' or 'MoS2'
        w(str): 'WSe2' or 'WS2'
        comparison: 'deg', 'pump', 'fluence', 'temp' legend parameters to show what is different between scans 
        fit_type_w: 'rd', 'rrd' (default) what type of fit to plut for W layer (rise_decay, rise_rise_decay)
        fit_type_m: 'rd' (default), 'rdd' what type of fit to plut for Mo layer (rise_decay, rise_decay_decay)
        norm(bool): scale all data to have maximum 1; default:False
        Bin(bool): plot binned data; default: False
        shorttime(bool): plot on range (-2,15) or full dataset; default: False (plot full dataset)
        ML(list): list of ML scan numbers to additionally plot; default: empty list
        error(bool): plot error; default:False
        title(str): title of plot; default: empty string
        plot_fit(bool): plot best fit line from default params in directory; default:False
        legend(bool): show legend; default: True
        inset(bool): add inset showing range of longtimes; default:False
    
    Outputs: 
        t vs. rms for m and w with m and w on same plot
    """
    from fitting_script import fit
    fig = plt.figure(figsize=(len(names)*5, 5), constrained_layout=True)
    # ax_list = [ax.subplot2grid((1, len(names)), (0,i)) for i, name in enumerate(names)]
    
    for i, name in enumerate(names): 
        ax = plt.subplot2grid((1, len(names)), (0, i))
        if inset: 
            axins = inset_axes(ax, width="40%", height="40%", loc=4, borderpad=1.5)
        data = scan(path+name)
        data.rms()
        compare_dict = {'deg':str(data.deg) + ' deg', 'pump':data.pump, 'fluence':data.fluence, 'temp':data.temp}
        if len(comparison) > 1: 
            i = 1
            while i < len(comparison):
                title += ' ' + compare_dict[comparison[i]]
                i+=1
        else: 
            title = compare_dict[comparison[0]]
        t0 = t0_dict[name]
        if not Bin: 
            delay = data.delay - t0 

        if data.bragg_info1.mat == m:
            if norm: 
                idx = np.argmax(data.rms1)
                vals = data.rms1[idx-4:idx+5]
                fac = np.nanmean(vals) 
            else:
                fac = 1
            if Bin: 
                temp = fit(path + name, plot=False)
                temp.Bin(1, 20, bin_limit='tot',  plot=False)
                data.delay = temp.binned_t
                data.rms1 = temp.binned_vals
                data.rms1_err = temp.binned_errs
            if error:     
                ax.errorbar(data.delay, data.rms1 * 1/fac, data.rms1_err * 1/fac, color='k', label=m)
                if inset: 
                    axins.errorbar(data.delay, data.rms1 * 1/fac, data.rms1_err * 1/fac, color='k', label=m)
            if plot_fit: 
                if fit_type_m == 'rd': 
                    params, err = fit_params_rd(name, 1)
                    y = rise_decay(delay, *params) * 1/fac
                    y[y < 0] = 0
                    ax.plot(delay - t0, y, color='k')
                    if inset: 
                        axins.plot(delay - t0, y, color='k')
                elif fit_type_m == 'rdd': 
                    params, err = fit_params_rdd(name, 1)
                    y = rise_decay_decay(delay, *params) * 1/fac
                    y[y< 0] = 0 
                    ax.plot(delay - t0, y, color='k')
                    if inset: 
                        axins.plot(delay - t0, y, color='k')
            ax.plot(delay, data.rms1 * 1/fac , marker='o', mew=2, markersize=4, linestyle='', color='k', label=m)
            if inset: 
                axins.plot(delay, data.rms1 * 1/fac , marker='o', mew=2, markersize=4, linestyle='', color='k', label=m)
            
        if data.bragg_info1.mat == w: 
            if norm: 
                idx = np.argmax(data.rms1)
                vals = data.rms1[idx-4:idx+5]
                fac = np.nanmean(vals) 
            else:
                fac = 1
            if Bin: 
                temp = fit(path + name, plot=False)
                temp.Bin(1, 20, bin_limit='tot',  plot=False)
                data.delay = temp.binned_t
                data.rms1 = temp.binned_vals
                data.rms1_err = temp.binned_errs
            if error:     
                ax.errorbar(data.delay, data.rms1 * 1/fac, data.rms1_err * 1/fac, color='r', label=w)
                if inset: 
                    axins.errorbar(data.delay, data.rms1 * 1/fac, data.rms1_err * 1/fac, color='r', label=w)
            if plot_fit: 
                if fit_type_w == 'rd': 
                    params, err = fit_params_rd(name, 1)
                    y[y < 0] = 0
                    ax.plot(delay - t0, y, color='r')
                    if inset: 
                        axins.plot(delay - t0, y, color='r')
                if fit_type_w == 'rrd': 
                    params, err = fit_params_rrd(name, 1)
                    y = rise_rise_decay(delay, *params) * 1/fac
                    y[y< 0] = 0 
                    ax.plot(delay - t0, y, color='r')
                    if inset: 
                        axins.plot(delay - t0, y, color='r')
            ax.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', color='r', label=w)
            if inset: 
                axins.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, markersize=4, linestyle = '', color='r', label=w)


        if data.bragg_info2.mat == m:
            if norm: 
                idx = np.argmax(data.rms2)
                vals = data.rms2[idx-4:idx+5]
                fac = np.nanmean(vals) 
            else:
                fac = 1
            if Bin:
                temp = fit(path + name, plot=False)
                temp.Bin(2, 20, bin_limit='tot',  plot=False)
                data.delay = temp.binned_t
                data.rms2 = temp.binned_vals
                data.rms2_err = temp.binned_errs
            if error:     
                ax.errorbar(data.delay, data.rms2 * 1/fac, data.rms2_err * 1/fac, color='k', label=m)
                if inset: 
                    axins.errorbar(data.delay, data.rms2 * 1/fac, data.rms2_err * 1/fac, color='k', label=m)
            if plot_fit: 
                if fit_type_m == 'rd': 
                    params, err = fit_params_rd(name, 2)
                    y = rise_decay(delay, *params) * 1/fac
                    y[y < 0] = 0
                    ax.plot(delay - t0, y, color= 'k')
                    if inset: 
                        axins.plot(delay - t0, y, color= 'k')
                if fit_type_m == 'rdd': 
                    params, err = fit_params_rdd(name, 2)
                    y = rise_rise_decay(delay, *params) * 1/fac
                    y[y< 0] = 0 
                    ax.plot(delay - t0, y, color='k')
                    if inset: 
                        axins.plot(delay - t0, y, color='k')
            ax.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', color='k', label=m)
            if inset: 
                axins.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', color='k', label=m)

        if data.bragg_info2.mat == w:
            if norm: 
                idx = np.argmax(data.rms2)
                vals = data.rms2[idx-4:idx+5]
                fac = np.nanmean(vals) 
            else:
                fac = 1
            if Bin: 
                temp = fit(path + name, plot=False)
                temp.Bin(2, 20, bin_limit='tot',  plot=False)
                data.delay = temp.binned_t
                data.rms2 = temp.binned_vals
                data.rms2_err = temp.binned_errs
            if error:     
                ax.errorbar(data.delay, data.rms2 * 1/fac, data.rms2_err * 1/fac, color='r', label=w)
                if inset: 
                    axins.errorbar(data.delay, data.rms2 * 1/fac, data.rms2_err * 1/fac, color='r', label=w)

            if plot_fit: 
                if fit_type_w == 'rd': 
                    params, err = fit_params_rd(name, 2)
                    y = rise_decay(delay, *params) * 1/fac
                    y[y < 0] = 0
                    ax.plot(delay - t0, y, color='r')
                    if inset: 
                        axins.plot(delay - t0, y, color='r')
                if fit_type_w == 'rrd': 
                    params, err = fit_params_rrd(name, 2)
                    y = rise_rise_decay(delay, *params) * 1/fac
                    y[y< 0] = 0 
                    ax.plot(delay - t0, y, color='r')
                    if inset: 
                        axins.plot(delay - t0, y, color='r')
            ax.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', color='r', label=w)
            if inset: 
                axins.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', color='r', label=w)

        if len(ML) > 0:
            for name in ML: 
                data = scan(path + name)
                data.rms()
                if data.bragg_info1.mat == 'WS2':
                    data.rms()
                    data.delay = data.delay[:-1]
                    data.rms1 = data.rms1[:-1]
                    data.rms1_err = data.rms1_err[:-1]
                delay = data.delay
                if norm: 
                    idx = np.argmax(data.rms1)
                    vals = data.rms1[idx-4:idx+5]
                    fac = np.nanmean(vals) 
                if Bin: 
                    temp = fit(path + name, plot=False)
                    temp.Bin(1, 20, bin_limit='tot',  plot=False)
                    data.delay = temp.binned_t
                    data.rms1 = temp.binned_vals
                    data.rms1_err = temp.binned_errs
                if data.bragg_info1.mat == w:
                    if error:     
                        ax.errorbar(data.delay, data.rms1 * 1/fac, data.rms1_err * 1/fac, color='b', label=data.bragg_info1.mat + ' ML', alpha=0.7, linestyle='--')
                        if inset: 
                            axins.errorbar(data.delay, data.rms1 * 1/fac, data.rms1_err * 1/fac, color='b', label=data.bragg_info1.mat + ' ML', alpha=0.7, linestyle='--')
                    if plot_fit: 
                        params, err = fit_params_rd(name, 1) 
                        if len(params) <=5 : 
                            y = rise_decay(delay, *params) * 1/fac
                            y[y < 0] = 0
                            ax.plot(delay - t0, y, color='b')
                            if inset: 
                                axins.plot(delay - t0, y, color='b')
                        else: 
                            y = rise_rise_decay(delay, *params) * 1/fac
                            y[y< 0] = 0 
                            ax.plot(delay - t0, y, color = 'b')
                            if inset: 
                                axins.plot(delay - t0, y, color = 'b')
                    ax.plot(delay, data.rms1 * 1/fac, marker='*', mew=2, markersize=4, linestyle='', color='b', label=w)
                    if inset: 
                        axins.plot(delay, data.rms1 * 1/fac, marker='*', mew=2, markersize=4, linestyle='', color='b', label=w)

                if data.bragg_info1.mat == m: 
                    if error:     
                        ax.errorbar(data.delay, data.rms1 * 1/fac, data.rms1_err * 1/fac, color='b', label=data.bragg_info1.mat + ' ML', alpha=0.7)
                        if inset: 
                            axins.errorbar(data.delay, data.rms1 * 1/fac, data.rms1_err * 1/fac, color='b', label=data.bragg_info1.mat + ' ML', alpha=0.7)
                    if plot_fit: 
                        params, err = fit_params_rd(name, 1) 
                        if len(params) <=5 : 
                            y = rise_decay(delay, *params) * 1/fac
                            y[y < 0] = 0
                            ax.plot(delay - t0, y, color = 'b')
                            if inset: 
                                axins.plot(delay - t0, y, color = 'b')
                        else: 
                            y = rise_rise_decay(delay, *params) * 1/fac
                            y[y< 0] = 0 
                            ax.plot(delay - t0, y, color = 'b')
                            if inset: 
                                axins.plot(delay - t0, y, color = 'b')
                    ax.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', color='b', label=m)
                    if inset: 
                        axins.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', color='b', label=m)
        if shorttime: 
            ax.set_xlim(-2,15)
        ax.set_title(title)
        ax.set_ylabel(r'$\Delta \langle u_{ip}^2 \rangle [\AA^2]$' , fontsize=15)
        ax.set_xlabel('Delay Time (ps)')
        if legend: 
            ax.legend()    


def plot_vals_mw(m, w, val, power, deg, names, fit_type_w='rrd', fit_type_m='rd', ML_names = []):
    """
    Plot specified best fit params using bragginfo1/2_fitparams.txt file in path 

    Inputs: 
        m(str): 'MoS2' or 'MoSe2'
        w(str): 'WS2' or 'WSe2' 
        val(str): parameter to plot, any parameter from rise_decay or rise_rise_decay
        power(bool): plot vs. fluence 
        deg(bool): plot vs. angle
        names(list): scan numbers to plot e.g 20211010_0006
        fit_type_w: 'rd', 'rrd' (default) what type of fit to plut for W layer (rise_decay, rise_rise_decay)
        fit_type_m: 'rd' (default), 'rdd' what type of fit to plut for Mo layer (rise_decay, rise_decay_decay)
        ML_names(list): list of ML scan numbers to additionally plot; default: empty list
    
    Outputs: 
        val vs. power or deg plot for m and w in separate plots 
    
    """

    labels = {'y0':0, 'A':1, 'a':2, 'x0':3, 'tau_1':4, 'tau_2':5, 'tau_3':6}
    labels_rd = {'y0':0, 'A':1, 'a':4, 'x0':2, 'tau_1':3, 'tau_3':4, 'tau_2':4}
    ylabel_dict = {'a':r'$\tau_2$ contribution', 'tau_1':r'$\tau_1$ [ps]', 'tau_2':r'$\tau_2$ [ps]', 'tau_3':r'$\tau_3$ [ps]'}
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(10,5))
    xrange = []
    for name in names: 
        data = scan(path + name)
        data.rms()
        if power: 
            x = int(data.fluence.replace('mj', ''))
            xrange.append(x)
        if deg: 
            x = data.deg
            xrange.append(x)
        if data.bragg_info1.mat == m:
            if fit_type_m == 'rdd':
                params, err = fit_params_rdd(name, 1)
                idx = labels[val]
            else: 
                params, err = fit_params_rd(name, 1)
                idx = labels_rd[val]
            y = params[idx]
            err = err[idx]
            ax1.scatter(x, y)
            ax1.errorbar(x, y, err)
        if data.bragg_info2.mat == m:
            if fit_type_m == 'rdd':
                params, err = fit_params_rdd(name, 2)
                idx = labels[val]
            else: 
                params, err = fit_params_rd(name, 2)
                idx = labels_rd[val]
            y = params[idx]
            err = err[idx]
            ax1.scatter(x, y)
            ax1.errorbar(x, y, err)
        if data.bragg_info1.mat == w:
            if fit_type_w == 'rrd':
                params, err = fit_params_rrd(name, 1)
                idx = labels[val]
            else: 
                params, err = fit_params_rd(name, 1)
                idx = labels_rd[val]
            y = params[idx]
            err = err[idx]
            ax2.scatter(x, y)
            ax2.errorbar(x, y, err)
        if data.bragg_info2.mat == w:
            if fit_type_w == 'rrd':
                params, err = fit_params_rrd(name, 2)
                idx = labels[val]
            else: 
                params, err = fit_params_rd(name, 2)
                idx = labels_rd[val]
            y = params[idx]
            err = err[idx]
            ax2.scatter(x, y)
            ax2.errorbar(x, y, err)

    points = np.arange(min(xrange) - 1, max(xrange) + 3)
    if val == 'tau_2' or val == 'a':
        pass
    else: 
        for name in ML_names: 
            data = scan(path + name)
            idx_rd = labels_rd[val]
            if data.bragg_info1.mat == m:
                idx = idx_rd
                params, err = fit_params_rd(name, 1)
                err = err[idx]
                y = params[idx]
                ax1.fill_between(points, np.repeat(y - err, len(points)), np.repeat(y + err, len(points)), color='k', alpha=0.3)
                ax1.hlines(y, min(xrange) - 1, max(xrange) + 2, color = 'k', linestyles='--')
            if data.bragg_info1.mat == w:
                idx = idx_rd
                params, err = fit_params_rd(name, 1)
                err = err[idx]
                y = params[idx]
                ax2.fill_between(points, np.repeat(y - err, len(points)), np.repeat(y + err, len(points)), color='k', alpha=0.3)
                ax2.hlines(y, min(xrange) - 1, max(xrange) + 2, color = 'k', linestyles='--')


    plt.tight_layout()
    if deg:
        ax1.set_xlabel('deg', size=20)
        ax2.set_xlabel('deg', size=20)
    if power:
        ax1.set_xlabel(r'mj / $cm^{2}$', size=20)
        ax2.set_xlabel(r'mj / $cm^{2}$', size=20)

    ax1.set_ylabel(ylabel_dict[val], size=25)
    mtitle = m.replace('2', '$_2$')
    wtitle = w.replace('2', '$_2$')
    ax1.set_title(mtitle)
    ax2.set_title(wtitle)


def plot_pretty_individual(names, m, w, comparison, ax1, ax2, fit_type_m = 'rd', fit_type_w='rrd', norm=False, Bin=False, ML_names=[], error=False, plot_fit=False, inset=False, idx=(5, 1), colors=None):
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
        colors: cycler for plot, default None goes to viridis cmap 
    
    Outputs: 
        t vs. rms for m and w with m and w on separate plots 
    """
    idx1, idx2 = idx
    if colors == None: 
        colors = []
        cmap = matplotlib.cm.get_cmap('viridis', 5)
        for i in range(cmap.N):
            rgba = cmap(i)
            colors.append(matplotlib.colors.rgb2hex(rgba))
    if plot_fit: 
        colors = [c for c in colors for _ in (0, 1)]
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


def plot_tau_individual(mat, compare, names, fit_type, val, ax, xrange, ML_names = []):
    """ 
    Plot specific param for a specific material in a set of scans

    Inputs: 
        mat(str): material; 'MoS2', 'MoSe2', 'WS2', 'WSe2'
        compare(str): value for x axis 'deg', 'temp', 'power'
        names(list): scan numbers to plot e.g 20211010_0006
        fit_type: 'rd' (rise_decay), 'rdd' (rise_decay_decay for Mo), 'rrd' (rise_rise_decay for W) which fitting model tau to plot
        ax (matplotlib.axes._subplots.AxesSubplot): subplot for plotting
        val: parameter to plot (a, tau_1, tau_2, tau_3)
        ML_names(list): list of ML scan numbers to additionally plot; default: empty list
    
    Outputs: 
        set of tau vs. compare value for specific material
    
    """
    labels = {'y0':0, 'A':1, 'a':2, 'x0':3, 'tau_1':4, 'tau_2':5, 'tau_3':6}
    labels_rd = {'y0':0, 'A':1, 'a':None, 'x0':2, 'tau_1':3, 'tau_2':None, 'tau_3':4}
    ylabel_dict = {'a':r'$\tau_2$ contribution', 'tau_1':r'$\tau_1$ [ps]', 'tau_2':r'$\tau_2$ [ps]', 'tau_3':r'$\tau_3$ [ps]'}

    colors = []
    cmap = matplotlib.cm.get_cmap('viridis', 5)
    for i in range(cmap.N):
        rgba = cmap(i)
        colors.append(matplotlib.colors.rgb2hex(rgba))
       
    new_cycler = (cycler(color=colors))
    plt.rc('axes', prop_cycle=new_cycler)

    if fit_type == 'rd':
        idx = labels_rd[val]
        for name in names:
            data = scan(path + name)
            if compare == 'power': 
                x = float(data.fluence.replace('mj', ''))
                ax.set_xlabel(rf'Fluence [mj cm$^{-2}$ ]')
            if compare == 'deg': 
                x = data.deg
                ax.set_xlabel(rf'Twist Angle [$\Delta \phi$]')
            if compare == 'temp':
                x = int(data.temp.replace('K', ''))
                ax.set_xlabel(rf'Temperature [K]')
            if data.bragg_info1.mat == mat:
                params, err = fit_params_rd(name, 1)
                y = params[idx]
                err = err[idx]
                ax.scatter(x, y)
                ax.errorbar(x, y, err)
            if data.bragg_info2.mat == mat:
                params, err = fit_params_rd(name, 2)
                y = params[idx]
                err = err[idx]
                ax.scatter(x, y)
                ax.errorbar(x, y, err)
        points = np.arange(min(xrange) - 1, max(xrange) + 3)
        for name in ML_names: 
            data = scan(path + name)
            idx = labels_rd[val]
            if data.bragg_info1.mat == mat:
                params, err = fit_params_rd(name, 1)
                err = err[idx]
                y = params[idx]
                ax.fill_between(points, np.repeat(y - err, len(points)), np.repeat(y + err, len(points)), color='k', alpha=0.3)
                ax.hlines(y, min(xrange) - 1, max(xrange) + 2, color = 'k', linestyles='--')
        title = mat.replace('2', '$_2$')
        ax.set_title(title)
        ax.set_ylabel(ylabel_dict[val])

    else:
        for name in names: 
            data = scan(path + name)
            data.rms()
            if compare == 'power': 
                x = float(data.fluence.replace('mj', ''))
                ax.set_xlabel(rf'power [$mj /cm^{2}$ ]')
            if compare == 'deg': 
                x = data.deg
                ax.set_xlabel(rf'Twist Angle [$\Delta \phi$]')
            if compare == 'temp':
                x = int(data.temp.replace('K', ''))
                ax.set_xlabel(rf'Temperature [K]')
            if data.bragg_info1.mat == mat:
                if 'Mo' in mat: 
                    params, err = fit_params_rdd(name, 1)
                    idx = labels[val]
                if 'W' in mat: 
                    params, err = fit_params_rrd(name, 1)
                    idx = labels[val]
                if idx != None: 
                    y = params[idx]
                    err = err[idx]
                    ax.scatter(x, y)
                    ax.errorbar(x, y, err)
            if data.bragg_info2.mat == mat:
                if 'Mo' in mat: 
                    params, err = fit_params_rdd(name, 2)
                    idx = labels[val]

                if 'W' in mat: 
                    params, err = fit_params_rrd(name, 2)
                    idx = labels[val]
                if idx != None: 
                    y = params[idx]
                    err = err[idx]
                    ax.scatter(x, y)
                    ax.errorbar(x, y, err)
        points = np.arange(min(xrange) - 1, max(xrange) + 3)
        for name in ML_names: 
            data = scan(path + name)
            idx = labels_rd[val]
            if data.bragg_info1.mat == mat and idx != None:
                params, err = fit_params_rd(name, 1)
                err = err[idx]
                y = params[idx]
                ax.fill_between(points, np.repeat(y - err, len(points)), np.repeat(y + err, len(points)), color='k', alpha=0.3)
                ax.hlines(y, min(xrange) - 1, max(xrange) + 2, color = 'k', linestyles='--')
        title = mat.replace('2', '$_2$')
        ax.set_title(title)
        ax.set_ylabel(ylabel_dict[val])
    plt.tight_layout()