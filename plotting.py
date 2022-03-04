#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from UED_Analysis_Functions import *
import matplotlib_specs
from fitting_script import *
from cycler import cycler
path = '/cds/home/a/amalyaj/Data/post_alignment/'
import matplotlib.pyplot as plt
import matplotlib 



def plot_pretty(names, m, w, comparison, shortM = False, shortW=False, norm=False, Bin=False, ML_names=[], error=False,
title = str(), plot_fit=False):
    """
    comparison = 'deg', 'pump', 'fluence', 'temp'
    compare = {'deg':data.deg, 'pump':data.pump, 'fluence':data.fluence, 'temp':data.temp}
    """
    new_cycler = (cycler(color=colors))
    plt.rc('axes', prop_cycle=new_cycler)
    mtitle = m + title 
    wtitle = w + title
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
        data.delay = data.delay - t0
        if not Bin: 
            delay = data.delay
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
                ax1.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, linestyle='', label=label)
                ax1.fill_between(delay, data.rms1 * 1/fac + data.rms1_err * 1/fac, data.rms1 * 1/fac - data.rms1_err * 1/fac, alpha=0.5)
            if not error: 
                ax1.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, linestyle='', label=label)

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
                ax1.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, linestyle='', label=label)
                ax1.fill_between(delay, data.rms2 * 1/fac + data.rms2_err * 1/fac, data.rms2 * 1/fac - data.rms2_err * 1/fac, alpha=0.5)
            if not error:
                ax1.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, linestyle='', label=label)

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
                ax2.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, linestyle='', label=label)
                ax2.fill_between(delay, data.rms1 * 1/fac + data.rms1_err * 1/fac, data.rms1 * 1/fac - data.rms1_err * 1/fac, alpha=0.5)
            if not error:
                ax2.plot(delay, data.rms1 * 1/fac, marker='o', mew=2, linestyle='', label=label)

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
                ax2.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, linestyle='', label=label)
                ax2.fill_between(delay, data.rms2 * 1/fac + data.rms2_err * 1/fac, data.rms2 * 1/fac - data.rms2_err * 1/fac, alpha=0.5)
            if not error:
                ax2.plot(delay, data.rms2 * 1/fac, marker='o', mew=2, linestyle='', label=label)

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
                if not error:
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
                if not error:
                    ax2.plot(delay, data.rms1 * 1/fac, label='Monolayer', color='k')


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