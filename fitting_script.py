#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import lmfit
from scipy import interpolate as interp
import matplotlib.pyplot as plt
from UED_Analysis_Functions import *
import matplotlib_specs 

a_dict = {'MoSe2': 3.32, 'WSe2': 3.32, 'MoS2':3.18, 'WS2':3.18}
bo_dict = {'order1' : [1,0,0], 'order2' : [2,-1,0], 
           'order3' : [2,0,0], 'order4' :[3,-1,0], 
           'order5' : [3,0,0], 'order6' : [4,-2,0], 
           'order7' : [4,-1,0], 'order8' : [4,0,0], 
           'order9' : [5,-2,0], 'order10' : [5,-1,0]}

class fit:
    """
    """
    def __init__(self, path, plot=True):
        """
        Fitting a scan from UEDU085
        Path: path of data file
        """
        
        self.scan = scan(path) 
        print('skipping orders:', self.scan.order_skip)
        self.scan.rms()
        self.scan.fit_log(plot=plot)
        delay = self.scan.delay
        delay = np.array(delay)
        self.delay = delay
        
        plt.savefig(path +'/linfit.png')
        if self.scan.type == 'HS':
            try: 
                p1 = np.array(pd.read_csv(os.path.join(path, 'bragginfo1_fitparams.txt'))['val'])
                p2 = np.array(pd.read_csv(os.path.join(path, 'bragginfo2_fitparams.txt'))['val'])
            except FileNotFoundError:
                mask = delay >=0.0
                p1, pcov = curve_fit(dblexppeak, delay[mask], u[mask], sigma=u_err[mask], absolute_sigma=True,
                bounds = ([0,0,-np.inf,0,0], [np.inf, np.inf, np.inf, np.inf, np.inf]))
                p2, pcov = curve_fit(dblexppeak, delay[mask], data.rms2[mask], sigma=data.rms2_err[mask], absolute_sigma=True,
                bounds = ([0,0,-np.inf,0,0], [np.inf, np.inf, np.inf, np.inf, np.inf]))
            self.p1 = p1
            self.p2 = p2
        if self.scan.type == 'ML':
            try:
                p1 = np.array(pd.read_csv(os.path.join(path, 'bragginfo1_fitparams.txt'))['val'])
            except FileNotFoundError:
                mask = delay >=0.0
                if self.scan.bragg_info1.mat == 'WS2':
                    delay = delay[:-1]
                    u = u[:-1]
                    u_err = u_err[:-1]
                    
                p1, pcov = curve_fit(dblexppeak, delay[mask], u[mask], sigma=u_err[mask], absolute_sigma=True,
                bounds = ([0,0,-np.inf,0,0], [np.inf, np.inf, np.inf, np.inf, np.inf]))
            self.p1 = p1

    def Bin(self, info, num_bins, t0=0.0, bin_limit='max'):
        """
        info: specify if we are looking at u or data.rms2
        t0: where fit begins, default 0.0
        bin_limit: specify where binning should end; default: 'max' --> bin to 5 points after the maximum
                 alternatively, 'tot' bin the entire raw data; or any idx chosen 
        """
        mask = self.scan.delay >= t0
        delay = self.delay 
        if info == 1: 
            u = self.scan.rms1
            u_err = self.scan.rms1_err
        else: 
            u = self.raw_data.rms2 
            u_err = self.raw_data.rms2_err
        
        if type(bin_limit) == str: 
            idx_dict = {'max':np.argmax(u)+5, 'tot':-1}
            idx_max = idx_dict[bin_limit]
        else: 
            idx_max = bin_limit
        print(idx_max)
        binned_rms = u[mask][-idx_max:]
        binned_rms_err = u_err[mask][-idx_max:]
        binned_delay = delay[mask][-idx_max:]

        binned_delay = np.array(binned_delay)
        sep = binned_rms.size/float(num_bins)*np.arange(1, num_bins)
        idx = sep.searchsorted(np.arange(binned_rms.size))

        binned_vals = []
        binned_errs = []
        binned_t = []

        if idx_max == -1:
            binned_vals = [u[0]]
            binned_errs = [u_err[0]]
            binned_t = [delay[0]]
        else:
            binned_vals = list(u[mask][:-idx_max])
            binned_errs = list(u_err[mask][:-idx_max])
            binned_t = list(delay[mask][:-idx_max])
            
            
        bins = np.arange(num_bins)
        for b in bins: 
            val = np.nanmean(binned_rms[np.where(idx == b)])
            err = 1/len(np.where(idx==b)[0]) * np.sqrt(np.sum(binned_rms_err[np.where(idx == b)]**2))
            binned_vals.append(val)
            binned_errs.append(err)
            binned_t.append(np.median(binned_delay[np.where(idx==b)]))

        pre = np.invert(mask)
        binned_vals.append(0)
        t0_err = 1/len(u_err[pre]) * np.sqrt(np.sum(u_err[pre]**2))
        binned_errs.append(t0_err)
        t0 = delay[mask][-1]
        binned_t.append(t0)

        plt.errorbar(delay, u, u_err, color='k', alpha=0.5, label='raw data')
        plt.errorbar(binned_t, binned_vals, binned_errs, color='b', label='binned data')
        plt.plot(binned_t[-1], binned_vals[-1], 'r*', markersize=10, label='t0')
        plt.plot(delay[-idx_max], u[-idx_max], 'r*', markersize=10, label='binning stopped here')
        if idx_max != -1: 
            plt.xlim(-2,15)
        plt.legend()
        self.binned_vals = binned_vals
        self.binned_errs = binned_errs
        self.binned_t = binned_t

    def scipy_fit(self, info, func, t0=0.0, shorttime=15, plot=True):
        """
        info: designate if data.rms1 or data.rms2
        func: rise_decay or rise_rise_decay
        shorttime: limits of x axis, if False, plot full data
        t0: where to begin fit 
        """
        delay = self.delay
        mask = delay >= t0
        if info == 1: 
            u = self.scan.rms1
            u_err = self.scan.rms1_err
            p0 = self.p1
            mat = self.scan.bragg_info1.mat
        else: 
            u = self.raw_data.rms2 
            u_err = self.raw_data.rms2_err
            p0 = self.p2
            mat = self.scan.bragg_info2.mat

        popt, pcov = curve_fit(func, delay[mask], u[mask],sigma = u_err[mask], absolute_sigma=True, 
                         p0=p0, bounds = ([0,0,-np.inf,0,0], [np.inf,np.inf,np.inf,np.inf,np.inf]))
        perr = np.sqrt(np.diag(pcov))
        self.sci_opt = popt
        self.sci_err = perr
        if len(popt) ==5: 
            print(f'{mat} scipy fit: y0:{popt[0]} +/- {perr[0]}, A:{popt[1]} +/- {perr[1]}, x0:{popt[2]} +/- {perr[2]}, t1:{popt[3]} +/- {perr[3]}, t2:{popt[4]} +/- {perr[4]}')
        else:
            print(f'{mat}, scipy fit: y0:{popt[0]} +/- {perr[0]}, A:{popt[1]} +/- {perr[1]}, x0:{popt[2]} +/- {perr[2]}, t1:{popt[3]} +/- {perr[3]}, t2:{popt[4]} +/- {perr[4]}, t3:{popt[5]} +/- {perr[5]}')
        if plot:
            plt.errorbar(delay, u, u_err, label='raw data')
#             return delay[mask], func(delay[mask], *popt)
            plt.plot(delay[mask], func(delay[mask], *popt), label='fit')
            plt.legend()
            if shorttime != False:  
                plt.xlim(-2,shorttime)


    def lm(self, info, func, scipy=True, varies=np.repeat(True, 6), t0=0.0, shorttime=15, plot=True):
        """
        scipy: fit with scipy first? 
        varies: if we want to vary any of the parameters
        """
        delay = self.scan.delay
        mask = delay >= t0

        self.scipy_fit(self, info, func, t0, shorttime, False)
        rdmodel = lmfit.Model(func)
        params = rdmodel.make_params()
        if scipy: 
            popt = self.sci_opt
        if info == 1: 
            u = self.scan.rms1
            u_err = self.scan.rms1_err
            if not scipy: 
                popt = self.p1
        else: 
            u = self.raw_data.rms2 
            u_err = self.raw_data.rms2_err
            if not scipy: 
                popt = self.p2 

        params['y0'] = lmfit.Parameter(name='y0', value=popt[0], vary = varies[0], min = 0, max = np.inf)
        params['A'] = lmfit.Parameter(name='A', value=popt[1], vary = varies[1], min = 0, max = np.inf)
        params['x0'] = lmfit.Parameter(name='x0', value=popt[2], vary=varies[2], min = -np.inf, max = np.inf)
        params['tau_1'] = lmfit.Parameter(name='tau_1', value=popt[3], vary=varies[3], min = 0, max = np.inf)
        params['tau_2'] = lmfit.Parameter(name='tau_2', value=popt[4], vary = varies[4], min=0, max=np.inf)
        if len(params) > 5: 
            params['tau_3'] = lmfit.Parameter(name='tau_3', value=popt[5], vary = varies[5], min=0, max=np.inf)
        result = rdmodel.fit(u[mask], params, x=u[mask], weights=1/u_err)
        print(result.fit_report())
        self.lm_opt = result
        
        if plot: 
            plt.errorbar(delay, u, u_err, label='raw data')
            plt.plot(delay[mask], self.lm_opt.best_fit, label='fit')
            plt.legend()
            if shorttime != False:  
                plt.xlim(-2,shorttime)

    def interp(self, info, func, points=1000, t0=0.0, shorttime=15, plot=True):
        delay = self.delay
        if info == 1: 
            u = self.scan.rms1
            u_err = self.scan.rms1_err
            p0 = self.p1
        else: 
            u = self.raw_data.rms2 
            u_err = self.raw_data.rms2_err
            p0 = self.p2
        
        f = interp.interp1d(delay, u)
        xnew = np.linspace(min(delay), max(delay), points)
        ynew = f(xnew)
        mask = xnew >= t0

        popt, pcov = curve_fit(func, xnew[mask], ynew[mask], p0 = p0, bounds = ([0,0,-np.inf,0,0], [np.inf,np.inf,np.inf,np.inf,np.inf]))
        perr = np.sqrt(np.diag(pcov))
        self.interp_opt = popt
        self.interp_err = perr

        if plot: 
            plt.errorbar(delay, u, u_err, label='raw data')
            plt.plot(delay[mask], func(delay[mask], *popt), label='fit')
            plt.legend()
            if shorttime != False:  
                plt.xlim(-2,shorttime)

