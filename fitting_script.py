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

# lattice parameters for relevant materials, from CrystalMaker
a_dict = {'MoSe2': 3.32, 'WSe2': 3.32, 'MoS2':3.18, 'WS2':3.18}

# an example plane for a specified Bragg order, used to calculate the reciprocal lattice vector to Bragg peaks in said order 
bo_dict = {'order1' : [1,0,0], 'order2' : [2,-1,0], 
           'order3' : [2,0,0], 'order4' :[3,-1,0], 
           'order5' : [3,0,0], 'order6' : [4,-2,0], 
           'order7' : [4,-1,0], 'order8' : [4,0,0], 
           'order9' : [5,-2,0], 'order10' : [5,-1,0]}

# values found for t_0 i.e where the excitation begins for a specific scan and where fitting should begin 
t0_dict = {'20211010_1021':-0.1, '20211009_2204':0.1, 
          '20211010_0006': 0.1, '20211009_1623':0.0,
          '20211008_0116':-0.1, '20211007_1521':-0.2, 
          '20211009_2129':0.2, '20211009_2101':0.2, 
          '20211009_1848':0.0, '20211008_1603':0.0, 
          '20211008_1436':0.0, '20211008_1423':0.0, 
          '20211007_1416':0.6, '20211007_1314':0.0, 
          '20211009_1124':0.2, '20211009_0337':0.1, 
          '20211009_0011':0.1, '20211008_2113':0.1, 
          '20211008:1906':0.1, '20211007_2246': 0.0, 
          '20211007_1753':-0.3, '20211007_0924':-0.2, 
          '20211007_0258':-0.1, '20211007_0102':0.0, 
          '20211006_2235':0.0, '20211006_2012':0.0,
          '20211006_1809':-0.2, '20211006_1615':-0.1,
          '20211010_0903':0.0, '20211010_1107':-0.1,
          '20211010_1021':0.0, '20211008_0116':0.0,
          '20211006_1328':-0.1, '20211006_1246':-0.2,
          '20211005_2304':-0.2, '20211008_1755':0.0, 
          '20211008_0939':0.0, '20211006_0731':-0.1, 
          '20211005_1759':0.0}



class fit:
    """
    Class to encompass extra tools for fitting rms atomic displacement data
    """
    def __init__(self, path, plot=True):
        """
        Initialize scan object on path
        
        Inputs: 
            path(str): path of processed data file that includes bragg peak intensity changes necessary for calculating rms atomic displacements
                and creating object of class "scan" 
            plot(bool): return plot and save to path 
        """
        
        self.scan = scan(path) 
        self.scan.rms()
        self.scan.fit_log(plot=plot)
        delay = self.scan.delay
        self.delay = np.array(delay)
        
        name = path[-13:]
        if name in t0_dict.keys():
            self.t0 = t0_dict[path[-13:]]
        else: 
            print('file not in dictionary, input t0') 
            t0 = input()
            self.t0 = float(t0)
        
        plt.savefig(path +'/linfit.png')
        if self.scan.type == 'HS':
            try: 
                p1 = np.array(pd.read_csv(os.path.join(path, 'bragginfo1_fitparams.txt'))['val'])
                p2 = np.array(pd.read_csv(os.path.join(path, 'bragginfo2_fitparams.txt'))['val'])
            except FileNotFoundError:
                mask = delay >=0.0
          
                p1, pcov = curve_fit(dblexppeak, delay[mask], self.scan.rms1[mask], sigma=self.scan.rms1_err[mask], absolute_sigma=True,
                bounds = ([0,0,-np.inf,0,0], [np.inf, np.inf, np.inf, np.inf, np.inf]))
                p2, pcov = curve_fit(dblexppeak, delay[mask], self.scan.rms2[mask], sigma=self.scan.rms2_err[mask], absolute_sigma=True,
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
                else: 
                    u = self.scan.rms1
                    u_err = self.scan.rms1_err
                    
                p1, pcov = curve_fit(dblexppeak, delay[mask], u[mask], sigma=u_err[mask], absolute_sigma=True,
                bounds = ([0,0,-np.inf,0,0], [np.inf, np.inf, np.inf, np.inf, np.inf]))
            self.p1 = p1

    def Bin(self, info, num_bins, mod_t0=False, bin_limit='max', plot=True):
        """
        Bin raw rms data (subset or full dataset) with specified number of bins. Binning starts at t0 and goes up to specified index. 
        
        Inputs: 
            info(int): 1 or 2, specify if we are looking at the first set of bragg peaks or the second (data.rms1, data.rms2)
            num_bins(int): number of bins to use 
            mod_t0: whether or not to change t0 from default value (self.t0), default False, if != False, must = float (new value for t0)
            bin_limit(str: 'max' or 'tot' or int)  specify where binning should end
                    if 'max' or 'tot': 
                        'max': bin to 5 points after maximum y value 
                        'tot': bin entire raw data 
                    if int: 
                        bin to this specified index 
            plot(bool): return plot of binned data, default True 

        Outputs: 
            self.binned_vals(np.array): y values of binned data, includes unbinned data if binning not done on entire range. 
                                        value calculated as mean of raw data points in bin 
            self.binned_errs(np.array): errors on y values of binned data, includes unbinned data if binning not done on entire range. 
                                        errors calculated from propogation of error of errors on raw data in bin 
            self.binned_t(np.array): x values of binned data, includes unbinned data if binning not done on entire range. 
                                     value calculated as median of t values in bin 
        """
        if mod_t0 == False: 
            t0 = self.t0 
        else: 
            t0 = mod_t0  
        mask = self.scan.delay >= t0
        delay = self.delay 
        if info == 1: 
            u = self.scan.rms1
            u_err = self.scan.rms1_err
        else: 
            u = self.scan.rms2 
            u_err = self.scan.rms2_err
        
        if type(bin_limit) == str: 
            idx_dict = {'max':np.argmax(u)+5, 'tot':-1}
            idx_max = idx_dict[bin_limit]
        else: 
            idx_max = bin_limit
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
        t0 = delay[mask][-1]
        t0_err = 1/len(u_err[pre]) * np.sqrt(np.sum(u_err[pre]**2))
        
        for i, t in enumerate(delay[pre]):
            binned_vals.append(np.nanmean(u[pre]))
            binned_errs.append(t0_err)
            binned_t.append(t)
        if plot == True: 
            plt.errorbar(delay, u, u_err, color='k', alpha=0.5, label='raw data')
            plt.errorbar(binned_t, binned_vals, binned_errs, color='b', label='binned data')
            plt.plot(binned_t[-1], binned_vals[-1], 'r*', markersize=10, label='t0')
            plt.plot(delay[-idx_max], u[-idx_max], 'r*', markersize=10, label='binning stopped here')
            if idx_max != -1: 
                plt.xlim(-2,15)
            plt.legend()
        
        self.binned_vals = np.array(binned_vals)
        self.binned_errs = np.array(binned_errs)
        self.binned_t = np.array(binned_t)
   
    def scipy_fit(self, info, func=rise_decay, mod_t0 = False, shorttime=15, plot=True):
        """
        Fit data using scipy.optimize.curve_fit. Initial guess is self.p1 or self.p2 i.e existing values in path for best fit parameters 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

        Inputs: 
            info(int): 1 or 2, specify if we are looking at the first set of bragg peaks or the second (data.rms1, data.rms2)
            func(function): function to fit with, rise_decay (default) or rise_rise_decay
            mod_t0: whether or not to change t0 from default value (self.t0), default False, if != False, must = float (new value for t0)
            shorttime: upper limit of x-axis, int (default = 15) or False to plot full x range
            plot(bool): return plot

        Outputs: 
            self.sci_opt(np.array): optimized parameters via scipy curve_fit  
            self.sci_err(np.array): errors on parameters via scipy curve_fit 
        """
        if mod_t0 == False: 
            t0 = self.t0 
        else: 
            t0 = mod_t0  

        delay = self.delay
        mask = delay >= t0
        if info == 1: 
            u = self.scan.rms1
            u_err = self.scan.rms1_err
            p0 = self.p1
            mat = self.scan.bragg_info1.mat
        else: 
            u = self.scan.rms2 
            u_err = self.scan.rms2_err
            p0 = self.p2
            mat = self.scan.bragg_info2.mat

        popt, pcov = curve_fit(func, delay[mask], u[mask],sigma = u_err[mask], absolute_sigma=True, 
                         p0=p0, bounds = ([0,0,-np.inf,0,0], [np.inf,np.inf,np.inf,np.inf,np.inf]))
        perr = np.sqrt(np.diag(pcov))
        self.sci_opt = popt
        self.sci_err = perr
        if len(popt) ==5: 
            print(f'{mat} scipy fit: y0:{round(popt[0],5)} +/- {round(perr[0],5)}, A:{round(popt[1],5)} +/- {round(perr[1],5)}, x0:{round(popt[2], 5)} +/- {round(perr[2],5)}, t1:{round(popt[3],5)} +/- {round(perr[3],5)}, t2:{round(popt[4],5)} +/- {round(perr[4],5)}')
        else:
            print(f'{mat}, scipy fit: y0:{round(popt[0],5)} +/- {round(perr[0],5)}, A:{round(popt[1],5)} +/- {round(perr[1],5)}, x0:{round(popt[2], 5)} +/- {round(perr[2],5)}, t1:{round(popt[3],5)} +/- {round(perr[3],5)}, t2:{round(popt[4],5)} +/- {round(perr[4],5)}, t3:{round(popt[5],5)}')
        if plot:
            plt.errorbar(delay, u, u_err, label='raw data')
#             return delay[mask], func(delay[mask], *popt)
            plt.plot(delay[mask], func(delay[mask], *popt), label='fit')
            plt.legend()
            if shorttime != False:  
                plt.xlim(-2,shorttime)


    def lm(self, info, func=rise_decay, scipy=True, num_bins=False, varies=np.repeat(True, 6), mod_t0=False, shorttime=15, plot=True):
        """
        Fit data using lmfit. Initial guess is self.p1 or self.p2 i.e existing values in path for best fit parameters OR parameters found
        fitting with scipy 
        https://lmfit.github.io/lmfit-py/intro.html
        
        Currently only set up for func=rise_decay; needs to be generalized to include rise_rise_decay 

        Inputs: 
            info(int): 1 or 2, specify if we are looking at the first set of bragg peaks or the second (data.rms1, data.rms2)
            func(function): function to fit with, rise_decay (default) or rise_rise_decay
            scipy(bool): whether or not to fit with scipy first to find initial guesses
            varies(np.array): np.array of len(parameters of func) bool values to specify which parameters to vary and which to hold 
            mod_t0: whether or not to change t0 from default value (self.t0), default False, if != False, must = float (new value for t0)
            shorttime: upper limit of x-axis, int (default = 15) or False to plot full x range
            plot(bool): return plot

        Outputs: 
            self.lm_opt: result of fit using lmfit 
        """
        if mod_t0 == False: 
            t0 = self.t0 
        else: 
            t0 = mod_t0  

        if scipy: 
            self.scipy_fit(info, func, t0, shorttime, False)
            popt = self.sci_opt
        
        else: 
            if info == 1:
                popt = self.p1
            else:
                popt = self.p2
        rdmodel = lmfit.Model(func)
        params = rdmodel.make_params()
        if num_bins != False: 
            self.Bin(info, num_bins=num_bins)
            u = self.binned_vals
            u_err = self.binned_errs
            delay = self.binned_t
            mask = delay >=t0
        else: 
            if info == 1: 
                u = self.scan.rms1
                u_err = self.scan.rms1_err
                if not scipy: 
                    popt = self.p1
            else: 
                u = self.scan.rms2 
                u_err = self.scan.rms2_err
                if not scipy: 
                    popt = self.p2 
            delay = self.scan.delay
            mask = delay >= t0

        params['y0'] = lmfit.Parameter(name='y0', value=popt[0], vary = varies[0], min = 0, max = np.inf)
        params['A'] = lmfit.Parameter(name='A', value=popt[1], vary = varies[1], min = 0, max = np.inf)
        params['x0'] = lmfit.Parameter(name='x0', value=popt[2], vary=varies[2], min = -np.inf, max = np.inf)
        params['tau_1'] = lmfit.Parameter(name='tau_1', value=popt[3], vary=varies[3], min = 0, max = np.inf)
        params['tau_2'] = lmfit.Parameter(name='tau_2', value=popt[4], vary = varies[4], min=0, max=np.inf)
        if len(params) > 5: 
            params['tau_3'] = lmfit.Parameter(name='tau_3', value=popt[5], vary = varies[5], min=0, max=np.inf)
        result = rdmodel.fit(u[mask], params, x=u[mask], weights=1/u_err[mask])
        print(result.fit_report())
        self.lm_opt = result
        
        if plot: 
            if num_bins == False: 
                plt.errorbar(delay, u, u_err, label='raw data')
            plt.plot(delay[mask], self.lm_opt.best_fit, label='fit')
            plt.legend()
            if shorttime != False:  
                plt.xlim(-2,shorttime)

    def interp(self, info, func=rise_decay, points=1000, mod_t0 = False, shorttime=15, plot=True):
        """
        Interpolate data to add more points and fit using scipy 

        Currently only set up for func=rise_decay; needs to be generalized to include rise_rise_decay 

        Inputs: 
            info(int): 1 or 2, specify if we are looking at the first set of bragg peaks or the second (data.rms1, data.rms2)
            func(function): function to fit with, rise_decay (default) or rise_rise_decay
            points(int): total number of points for xrange 
            mod_t0: whether or not to change t0 from default value (self.t0), default False, if != False, must = float (new value for t0)
            shorttime: upper limit of x-axis, int (default = 15) or False to plot full x range
            plot(bool): return plot

        Outputs: 
            self.interp_opt: optimized parameters using curve_fit on interpolated data 
            self.interp_err: errors on parameters found using curve_fit on itnerpolated data 
        """
        if mod_t0 == False: 
            t0 = self.t0 
        else: 
            t0 = mod_t0  

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

