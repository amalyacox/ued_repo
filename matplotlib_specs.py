#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt 
import matplotlib

default_text_size=15
font_times = {'family':'DejaVu Sans'}

plt.rc('font', size=default_text_size)          # controls default text sizes
plt.rc('axes', titlesize=default_text_size)     # fontsize of the axes title
plt.rc('axes', labelsize=default_text_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=default_text_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=default_text_size)    # fontsize of the tick labels
plt.rc('figure', titlesize=default_text_size)  # fontsize of the figure title
plt.rc('font', **font_times)
plt.rc('figure', figsize=(8,6))
plt.rc('legend', fontsize=15, frameon=False)
plt.rc('lines', linewidth=2)
plt.rc('axes', linewidth=1.5)
lw = 2 #default line width
ms = 15
plt.rcParams.update({'figure.max_open_warning': 0})

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
matplotlib.rcParams['axes.linewidth']  = 1.5
plt.rcParams['lines.markersize'] = 4