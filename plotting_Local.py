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

path = '/home/amalyajohnson/Research/UED_U085/Processed_Data/'

def rise_rise_decay(x, y0, A, a, x0, tau_1, tau_3, tau_2): 
    """
    Exponential rise & decay convolution with sigma = 0.15; instrument resolution 
    Inputs: x, y0, A, x0, tau_1, tau_2. x: independent paramater for fit 
    Outputs: exponential rise & decay curve
    """
    return y0 + A*(-(1-a)*np.exp(-(x-x0)/tau_1 + (0.15**2)/(2*tau_1**2)) + -a*np.exp(-(x-x0)/tau_3 + (0.15**2)/(2*tau_3**2)) + np.exp(-(x-x0)/tau_2 + (0.15**2)/(2*tau_2**2)))

def rise_decay_decay(x, y0, A, a, x0, tau_1, tau_3, tau_2): 
    """
    Exponential rise & decay convolution with sigma = 0.15; instrument resolution 
    Inputs: x, y0, A, x0, tau_1, tau_2. x: independent paramater for fit 
    Outputs: exponential rise & decay curve
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

def plot_pretty(names, m, w, comparison, shortM = False, shortW=False, norm=False, Bin=False, ML_names=[], error=False,
title = str(), plot_fit=False, legend=True):
    """
    comparison = 'deg', 'pump', 'fluence', 'temp'
    compare = {'deg':data.deg, 'pump':data.pump, 'fluence':data.fluence, 'temp':data.temp}
    """
    colors = []
    cmap = matplotlib.cm.get_cmap('viridis', 5)
    for i in range(cmap.N):
        rgba = cmap(i)
        colors.append(matplotlib.colors.rgb2hex(rgba))
#     colors = colors[::-1]    
    if plot_fit: 
        colors = [c for c in colors for _ in (0, 1)]
        new_cycler = (cycler(color=colors))
    else: 
        new_cycler = (cycler(color=colors))
    plt.rc('axes', prop_cycle=new_cycler)
    mtitle = m.replace('2', '$_2$') + title 
    wtitle = w.replace('2', '$_2$') + title
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(10,5))
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
        if not Bin: 
            delay = data.delay - t0
        if data.bragg_info1.mat == m:
            if norm: 
                idx = np.argmax(data.rms1)
                vals = data.rms1[idx-5:idx+1]
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
                delay = delay - t0
                mask = delay >= 0 
                params, err = fit_params(name, 1) 
                if len(params) <=5 : 
                    ax1.plot(delay[mask], rise_decay(delay[mask], *params) * 1/fac)
                else: 
                    ax1.plot(delay[mask], rise_rise_decay(delay[mask], *params) * 1/fac)        
            if not error or plot_fit: 
                ax1.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', label=label)

        if data.bragg_info2.mat == m:
            if norm: 
                idx = np.argmax(data.rms2)
                vals = data.rms2[idx-5:idx+1]
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
                params, err = fit_params(name, 2) 
                if len(params) <=5 : 
                    ax1.plot(delay[mask], rise_decay(delay[mask], *params) * 1/fac)
                else: 
                    ax1.plot(delay[mask], rise_rise_decay(delay[mask], *params) * 1/fac)   
            if not error or plot_fit:
                ax1.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', label=label)

        if data.bragg_info1.mat == w:
            if norm: 
                idx = np.argmax(data.rms1)
                vals = data.rms1[idx-5:idx+1]
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
                params, err = fit_params(name, 1) 
                if len(params) <=5 : 
                    ax2.plot(delay[mask], rise_decay(delay[mask], *params) * 1/fac)
                else: 
                    ax2.plot(delay[mask], rise_rise_decay(delay[mask], *params) * 1/fac)   
            if not error or plot_fit:
                ax2.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', label=label)

        if data.bragg_info2.mat == w:
            if norm: 
                idx = np.argmax(data.rms2)
                vals = data.rms2[idx-5:idx+1]
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
                params, err = fit_params(name, 2) 
                if len(params) <=5 : 
                    ax2.plot(delay[mask], rise_decay(delay[mask], *params) * 1/fac)
                else: 
                    ax2.plot(delay[mask], rise_rise_decay(delay[mask], *params) * 1/fac)   
            if not error or plot_fit:
                ax2.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, markersize=4, linestyle='', label=label)

    if len(ML_names) > 0:     
        for name in ML_names:
            data = scan(path + name)
            data.rms()
            if data.bragg_info1.mat == m:
                if norm: 
                    idx = np.argmax(data.rms1)
                    vals = data.rms1[idx-5:idx+1]
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
                    ax1.plot(delay, data.rms1 * 1/fac, label='Monolayer', color='k')
                    ax1.fill_between(delay, data.rms1 * 1/fac + data.rms1_err * 1/fac, data.rms1 * 1/fac - data.rms1_err * 1/fac, color='k', alpha=0.5)
                if plot_fit: 
                    mask = delay >= 0 
                    params, err = fit_params(name, 1) 
                    if len(params) <=5 : 
                        ax1.plot(delay[mask], rise_decay(delay[mask], *params) * 1/fac)
                    else: 
                        ax1.plot(delay[mask], rise_rise_decay(delay[mask], *params) * 1/fac)   
                if not error or plot_fit:
                    ax1.plot(delay, data.rms1 * 1/fac, label='Monolayer', color='k')

            if data.bragg_info1.mat == w: 
                if w == 'WS2':
                    data.rms()
                    data.delay = data.delay[:-1]
                    data.rms1 = data.rms1[:-1]
                    data.rms1_err = data.rms1_err[:-1]
                if norm: 
                    idx = np.argmax(data.rms1)
                    vals = data.rms1[idx-5:idx+1]
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
                    ax2.plot(delay, data.rms1 * 1/fac, label='Monolayer', color='k')
                    ax2.fill_between(delay, data.rms1 * 1/fac + data.rms1_err * 1/fac, data.rms1 * 1/fac - data.rms1_err * 1/fac, color='k', alpha=0.3)
                if plot_fit: 
                    mask = delay >= 0 
                    params, err = fit_params(name, 1) 
                    if len(params) <=5 : 
                        ax2.plot(delay[mask], rise_decay(delay[mask], *params) * 1/fac)
                    else: 
                        ax2.plot(delay[mask], rise_rise_decay(delay[mask], *params) * 1/fac)   
                if not error or plot_fit:
                    ax2.plot(delay, data.rms1 * 1/fac, label='Monolayer', color='k')

    if legend:
        ax1.legend()
        ax2.legend()
    if shortM !=False:
        ax1.set_xlim(shortM)
    if shortW !=False:
        ax2.set_xlim(shortW)
    ax1.set_ylabel(r'$\Delta \langle u_{ip}^2 \rangle [\AA^2]$' , fontsize=15)
    ax1.set_xlabel('Delay Time (ps)')
    ax2.set_xlabel('Delay Time (ps)')
    ax1.set_title(mtitle)
    ax2.set_title(wtitle)
    plt.tight_layout()

def plot_tau_single(mat, power, deg, names, ML_names = []):
    labels = {'y0':0, 'A':1, 'a':2, 'x0':3, 'tau_1':4, 'tau_2':5, 'tau_3':6}
    labels_rd = {'y0':0, 'A':1, 'a':None, 'x0':2, 'tau_1':3, 'tau_2':None, 'tau_3':4}
    
    if (scan(path + names[0]).bragg_info1.mat == mat and len(fit_params(names[0], 1)[0]) == 5) or (scan(path + names[0]).bragg_info2.mat == mat and len(fit_params(names[0], 2)[0]) == 5):
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(12,6))
        for val, ax in zip(['tau_1', 'tau_3'], [ax1, ax2]):
            xrange = []
            for name in names:
                idx = labels_rd[val]
                print('using rise decay for Mo')
                data = scan(path + name)
                if power: 
                    x = int(data.fluence.replace('mj', ''))
                    xrange.append(x)
                    ax.set_xlabel(rf'power [$mj /cm^{2}$ ]')
                if deg: 
                    x = data.deg
                    xrange.append(x)
                    ax.set_xlabel(rf'Twist Angle [$\Delta \phi$]')
                if data.bragg_info1.mat == mat:
                    params, err = fit_params(name, 1)
                    y = params[idx]
                    err = err[idx]
                    ax.scatter(x, y)
                    ax.errorbar(x, y, err)
                if data.bragg_info2.mat == mat:
                    params, err = fit_params(name, 2)
                    y = params[idx]
                    err = err[idx]
                    ax.scatter(x, y)
                    ax.errorbar(x, y, err)
            for name in ML_names: 
                data = scan(path + name)
                idx = labels_rd[val]
                if data.bragg_info1.mat == mat:
                    params, err = fit_params(name, 1)
                    err = err[idx]
                    y = params[idx]
                    ax.fill_between(points, np.repeat(y - err, len(points)), np.repeat(y + err, len(points)), color='k', alpha=0.3)
                    ax.hlines(y, min(xrange) - 1, max(xrange) + 2, color = 'k', linestyles='--')

            ax.set_title(mat)
            ax.set_ylabel(val + ' [ps]')
        plt.tight_layout()

    else:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, sharey=False, figsize=(20,5))
        for val, ax in zip(['a', 'tau_1', 'tau_2', 'tau_3'], [ax1, ax2, ax3, ax4]):
            xrange = []
            for name in names: 
                idx = labels[val]
                data = scan(path + name)
                data.rms()
                if power: 
                    x = int(data.fluence.replace('mj', ''))
                    xrange.append(x)
                    ax.set_xlabel(rf'power [$mj /cm^{2}$ ]')
                if deg: 
                    x = data.deg
                    xrange.append(x)
                    ax.set_xlabel(rf'Twist Angle [$\Delta \phi$]')
                if data.bragg_info1.mat == mat:
                    params, err = fit_params(name, 1)
                    y = params[idx]
                    err = err[idx]
                    ax.scatter(x, y)
                    ax.errorbar(x, y, err)
                if data.bragg_info2.mat == mat:
                    params, err = fit_params(name, 2)
                    y = params[idx]
                    err = err[idx]
                    ax.scatter(x, y)
                    ax.errorbar(x, y, err)
            for name in ML_names: 
                data = scan(path + name)
                idx = labels_rd[val]
                if data.bragg_info1.mat == mat and idx != None:
                    params, err = fit_params(name, 1)
                    err = err[idx]
                    y = params[idx]
                    ax.fill_between(points, np.repeat(y - err, len(points)), np.repeat(y + err, len(points)), color='k', alpha=0.3)
                    ax.hlines(y, min(xrange) - 1, max(xrange) + 2, color = 'k', linestyles='--')
            ax.set_title(mat)
            ax.set_ylabel(val)
        plt.tight_layout()

def plot_m_w(names, m, w, comparison, norm=False, Bin=True, shorttime=False, ML=[], title=str()):
    for name in names: 
        data = scan(path+name)
        data.rms()
        compare_dict = {'deg':str(data.deg) + ' deg', 'pump':data.pump, 'fluence':data.fluence, 'temp':data.temp}
        if len(comparison) > 1: 
            i = 1
            while i < len(comparison):
                title += ' ' + compare_dict[comparison[i]]
                i+=1
        plt.figure(figsize=(5,5))
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
            plt.errorbar(data.delay, data.rms1 * 1/fac, data.rms1_err * 1/fac, color='k', label=m)
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
            plt.errorbar(data.delay, data.rms1 * 1/fac, data.rms1_err * 1/fac, color='r', label=w)
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
            plt.errorbar(data.delay, data.rms2 * 1/fac, data.rms2_err * 1/fac, color='k', label=m)
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
            plt.errorbar(data.delay, data.rms2 * 1/fac, data.rms2_err * 1/fac, color='r', label=w)
        if len(ML) > 0:
            for name in ML: 
                data = scan(path + name)
                data.rms()
                if data.bragg_info1.mat == 'WS2':
                    data.rms()
                    data.delay = data.delay[:-1]
                    data.rms1 = data.rms1[:-1]
                    data.rms1_err = data.rms1_err[:-1]
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
                    plt.errorbar(data.delay, data.rms1 * 1/fac, data.rms1_err * 1/fac, color='b', label=data.bragg_info1.mat + ' ML', alpha=0.7, linestyle='--')
                if data.bragg_info1.mat == m: 
                    plt.errorbar(data.delay, data.rms1 * 1/fac, data.rms1_err * 1/fac, color='b', label=data.bragg_info1.mat + ' ML', alpha=0.7)
        if shorttime: 
            plt.xlim(-2,15)
        plt.title(title)
        plt.ylabel(r'$\Delta \langle u_{ip}^2 \rangle [\AA^2]$' , fontsize=15)
        plt.xlabel('Delay Time (ps)')
        plt.legend()