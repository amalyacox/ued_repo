#!/usr/bin/env python
# coding: utf-8
import shutil
print(shutil.which('python'))
import numpy as np
import pandas as pdF
from UED_Analysis_Functions import *
import matplotlib_specs
from fitting_script import *
# from cycler import cycler
path = '/cds/home/a/amalyaj/Data/post_alignment/'
import matplotlib.pyplot as plt
import matplotlib 

colors = []
cmap = matplotlib.cm.get_cmap('viridis', 5)
for i in range(cmap.N):
    rgba = cmap(i)
    colors.append(matplotlib.colors.rgb2hex(rgba))


def fit_params(name, info): 
    path = '/cds/home/a/amalyaj/Data/post_alignment/'
    fn = path + name
    data = scan(fn)
    if info ==1:
        params = pd.read_csv(f'{fn}/bragginfo1_fitparams.txt')['val']
        err = pd.read_csv(f'{fn}/bragginfo1_fitparams.txt')['err']
    if info ==2: 
        params = pd.read_csv(f'{fn}/bragginfo2_fitparams.txt')['val']
        err = pd.read_csv(f'{fn}/bragginfo2_fitparams.txt')['err']
    return params, err 

def plot_pretty(names, m, w, comparison, shortM = False, shortW=False, norm=False, Bin=False, ML_names=[], error=False,
title = str(), plot_fit=False, legend=True):
    """
    comparison = 'deg', 'pump', 'fluence', 'temp'
    compare = {'deg':data.deg, 'pump':data.pump, 'fluence':data.fluence, 'temp':data.temp}
    """
    colors = []
    cmap = matplotlib.cm.get_cmap('viridis', 4)
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
                vals = data.rms1[idx-4:idx+5]
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
                vals = data.rms2[idx-4:idx+5]
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
                vals = data.rms1[idx-4:idx+5]
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
                vals = data.rms2[idx-4:idx+5]
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
                    vals = data.rms1[idx-4:idx+5]
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
                    vals = data.rms1[idx-4:idx+5]
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
    

def plot_m_w(names, m, w, comparison, norm=False, Bin=True, shorttime=False, ML=[], title=str()):
    for name in names: 
        data = scan(path+name)
        data.rms()
        compare_dict = {'deg':str(data.deg) + ' deg', 'pump':data.pump, 'fluence':data.fluence, 'temp':data.temp}
        if title != None:
            title = title + compare_dict[comparison[0]]
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
    
    
