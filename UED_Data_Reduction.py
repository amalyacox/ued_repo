import numpy as np

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

def mask_peaks(ind_r1:int, ind_r2:int, order:int, roll1:int, roll2:int, plot:bool=False):
    """
    Create polar bragg peak and bragg peak background masks
    Inputs: ind_r1, ind_r2: the slice of the polar image of interest for this order 
            order: order of interest
            roll1, roll2: how far to roll background forwards or backgrounds
            plot: if want to plot every individual peak 
    Outputs: polar_img_seg: full polar image of this section 
             r_seg: r segment of polar image of this section 
    
    """
    slice_r_seg = slice(ind_r1, ind_r2)
    r_seg = np.arange(ind_r1, ind_r2)
    
    m_total = 0
    global bragg_info1
    global bragg_info2 
    global img_mean_rr
    
    bragg_info1['bragg_index_order'+str(order)+'_pm'] = []
    bragg_info1['bragg_index_order'+str(order)+'_bm'] = []
    bragg_info1['bragg_index_order'+str(order)+'_slice_r_seg'] = slice_r_seg
    bragg_info1['bragg_index_order'+str(order)+'_r_axis'] = r_seg

    bragg_info2['bragg_index_order'+str(order)+'_pm'] = []
    bragg_info2['bragg_index_order'+str(order)+'_bm'] = []
    bragg_info2['bragg_index_order'+str(order)+'_slice_r_seg'] = slice_r_seg
    bragg_info2['bragg_index_order'+str(order)+'_r_axis'] = r_seg
    
    plot_info1_x = np.array([])
    plot_info1_y = np.array([])
    plot_info1_txt = np.array([])

    plot_info2_x = np.array([])
    plot_info2_y = np.array([])
    plot_info2_txt = np.array([])
    
    for ind, ind2 in zip(bragg_info1['bragg_index_order'+str(order)], bragg_info2['bragg_index_order'+str(order)]): 
        cm1_seg = bragg_info1['bragg_mask_set'][ind]
        pm1_seg, ptSettings = polarTransform.convertToPolarImage(cm1_seg, center=bragg_info1['centroid_main_beam'][::-1])
        pm1_seg = pm1_seg>0.1
        pm1_seg = pm1_seg[:, slice_r_seg]
        polarImg_seg = polarImage[:, slice_r_seg]
        bm1_seg = np.roll(pm1_seg, roll1, axis=0)

        cm2_seg = bragg_info2['bragg_mask_set'][ind2]
        pm2_seg, ptSettings = polarTransform.convertToPolarImage(cm2_seg, center=bragg_info2['centroid_main_beam'][::-1])
        pm2_seg = pm2_seg>0.05
        pm2_seg = pm2_seg[:, slice_r_seg]
        bm2_seg = np.roll(pm2_seg, -roll2, axis=0)
        
        if plot:

            fig = plt.figure(constrained_layout=True, figsize=(8,8))
            gs = fig.add_gridspec(3, 2)

            ax1 = fig.add_subplot(gs[0, :])
            ax1.pcolormesh(t_axis, r_seg, np.log(polarImg_seg*(pm1_seg+bm1_seg)).T)
            ax1.set_title(bragg_info1['bragg_mask_order'][ind]+' peak set 1')

            ax2 = fig.add_subplot(gs[1, :])
            ax2.pcolormesh(t_axis, r_seg, np.log(polarImg_seg*(pm2_seg+bm2_seg)).T)
            ax2.set_title(bragg_info2['bragg_mask_order'][ind2]+' peak set 2')

            ax3 = fig.add_subplot(gs[2, 0])
            ax3.imshow((img_mean_rr*cm1_seg))
            ax3.set_xlim(bragg_info1['centroids_bragg_all'][ind][1]-30, bragg_info1['centroids_bragg_all'][ind][1]+30)
            ax3.set_ylim(bragg_info1['centroids_bragg_all'][ind][0]-30, bragg_info1['centroids_bragg_all'][ind][0]+30)
            ax3.set_title(bragg_info1['bragg_mask_order'][ind]+' peak set 1')

            ax4 = fig.add_subplot(gs[2, 1])
            ax4.imshow((img_mean_rr*cm2_seg))
            ax4.set_xlim(bragg_info2['centroids_bragg_all'][ind2][1]-30, bragg_info2['centroids_bragg_all'][ind2][1]+30)
            ax4.set_ylim(bragg_info2['centroids_bragg_all'][ind2][0]-30, bragg_info2['centroids_bragg_all'][ind2][0]+30)
            ax4.set_title(bragg_info2['bragg_mask_order'][ind2]+' peak set 2')

        bragg_info1['bragg_index_order'+str(order)+'_pm'].append(pm1_seg.astype(int))
        bragg_info1['bragg_index_order'+str(order)+'_bm'].append(bm1_seg.astype(int))

        bragg_info2['bragg_index_order'+str(order)+'_pm'].append(pm2_seg.astype(int))
        bragg_info2['bragg_index_order'+str(order)+'_bm'].append(bm2_seg.astype(int))


        m_total += (pm1_seg+bm1_seg+pm2_seg+bm2_seg)

        x, y = center_of_mass(pm1_seg)
        text = str(ind)
        plot_info1_x = np.append(plot_info1_x, x)
        plot_info1_y = np.append(plot_info1_y, y)
        plot_info1_txt =  np.append(plot_info1_txt, text)

        x, y = center_of_mass(pm2_seg)
        text = str(ind2)
        plot_info2_x = np.append(plot_info2_x, x)
        plot_info2_y = np.append(plot_info2_y, y)
        plot_info2_txt =  np.append(plot_info2_txt, text)
        
        img = np.log(polarImg_seg*(m_total)).T
        y_fac = (max(r_seg) - min(r_seg))/(img.shape[0])
        x_fac = max(t_axis)/(img.shape[1])
        y = plot_info1_y * y_fac + min(r_seg)
        x = plot_info1_x * x_fac

    plt.figure()
    plt.subplot(2,1,1)
    plt.pcolormesh(t_axis, r_seg, np.log(polarImg_seg*(m_total)).T)
    print(polarImg_seg.shape, m_total.shape)
    plt.title('Masked peaks')
    plt.subplot(2,1,2)
    plt.pcolormesh(t_axis, r_seg, np.log(polarImg_seg).T)
    t_pro = np.mean(polarImg_seg, axis=1)
    t_pro = t_pro / np.max(t_pro) * polarImg_seg.shape[1]*0.3 + np.min(r_seg)
    plt.plot(t_axis, t_pro, 'w-', lw=1.5)
    for ind, val in enumerate(x):
        plt.text(val, y[ind]+10, plot_info1_txt[ind], color='w')
        
    return r_seg, polarImg_seg

def cut_peaks(vals_1:list, vals_2:list, order:int, overwrite:bool=False):
    """
    After visually inspecting the masks, choose indices possibly want to cut
    inputs: vals_1, vals_2: selected indices to cut in bragg_info1, bragg_info2
            order: order of interest
            overwrite: default False; if True: overwrite bragg_info['bragg_index_order'] with ind_info1, ind_info2
    outputs: plot without selected peaks 

    """
        
    global bragg_info1
    global bragg_info2 
    global polarImg_seg
    global r_seg
    
    # cutting selected peaks out of bragg_index_order"x"
    ind_info1 = [v for v in bragg_info1['bragg_index_order'+str(order)] if v not in vals_1]
    ind_info2 = [v for v in bragg_info2['bragg_index_order'+str(order)] if v not in vals_2]
    
    # Find the indices (relative to the bragg index list) of the peaks we want to cut 
    cut_info1 = [bragg_info1['bragg_index_order'+str(order)].index(v) for v in vals_1]
    cut_info2 = [bragg_info2['bragg_index_order'+str(order)].index(v) for v in vals_2]
    
    # cut these peaks out of pm and bm but don't actually change the values 
    pm1_new = [v for i, v in enumerate(bragg_info1['bragg_index_order'+str(order)+'_pm']) if i not in cut_info1]
    bm1_new = [v for i, v in enumerate(bragg_info1['bragg_index_order'+str(order)+'_bm']) if i not in cut_info1]

    pm2_new = [v for i, v in enumerate(bragg_info2['bragg_index_order'+str(order)+'_pm']) if i not in cut_info2]
    bm2_new = [v for i, v in enumerate(bragg_info2['bragg_index_order'+str(order)+'_bm']) if i not in cut_info2]
    
    tot = sum(pm1_new) + sum(bm1_new) + sum(pm2_new) + sum(bm2_new)
    plt.subplot(211)
    plt.pcolormesh(t_axis, r_seg, np.log(polarImg_seg*(tot)).T)
    plt.subplot(212)
    plt.pcolormesh(t_axis, r_seg, np.log(polarImg_seg).T)
    
    if overwrite: 
        print('OVERWRITING bragg_info1/2[\'bragg_index_order'+str(order)+']')
        print('OVERWRITING bragg_info1/2[\'bragg_index_order'+str(order)+'_pm/bm]')
        bragg_info1['bragg_index_order'+str(order)] = ind_info1
        bragg_info2['bragg_index_order'+str(order)] = ind_info2
        
        bragg_info1['bragg_index_order'+str(order)+'_pm'] = pm1_new
        bragg_info1['bragg_index_order'+str(order)+'_bm'] = bm1_new
        bragg_info2['bragg_index_order'+str(order)+'_pm'] = pm2_new
        bragg_info2['bragg_index_order'+str(order)+'_bm'] = bm2_new