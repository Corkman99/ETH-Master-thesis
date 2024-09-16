"""
Plots for Importance Measures

"""
import numpy as np
import xarray as xr

import time

from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize
import cartopy.crs as ccrs

from dask.distributed import Client, LocalCluster

## File load ect
indir = '/net/litho/atmosdyn/roethlim/data/lastvar/era5/upload/TX1day/data/figures/'
infile = 'TX1day_decomposition_era5_v10_final.nc'
outdir = 'plots/final/'

xr_in = xr.open_dataset(indir + '/' + infile,
                        chunks = {'years' : 42,'lon': 181, 'lat': 91}).mean(dim='lev',skipna=True)
# drop first year because of NaN
xr_in = xr_in.drop_sel(years = 1979)

def permutation(dftanom,dfadv,dfadiab,dfdiab):
    df = np.stack((dfadv,dfadiab,dfdiab),axis=1)
    mod = LinearRegression(fit_intercept=False).fit(df,dftanom)
    importance = permutation_importance(mod, df, dftanom,
                                        scoring='explained_variance',
                                        n_repeats=20,
                                        n_jobs=1,
                                        random_state=16)['importances_mean']
    sorted_ind = np.flip(np.argsort(importance))
    vars = ['Adv','Adiab','Diab']
    dom = vars[sorted_ind[0]]
    mapping = {'Adv':1/6,'Adiab':3/6,'Diab':5/6}
    return mapping.get(dom,np.nan), importance[sorted_ind[0]], importance[sorted_ind[1]], importance[sorted_ind[2]]

def permutation_masked(dftanom,dfadv,dfadiab,dfdiab):
    df = np.stack((dfadv,dfadiab,dfdiab),axis=1)
    corrs = np.corrcoef(df,rowvar=False)
    if  any(np.abs([corrs[0,1],corrs[0,2],corrs[1,2]]) > 0.8): # case where correlation hinders interpretation
        return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
    else:
        mod = LinearRegression(fit_intercept=False).fit(df,dftanom)
        importance = permutation_importance(mod, df, dftanom,
                                        scoring='r2',
                                        n_repeats=20,
                                        n_jobs=1,
                                        random_state=16)
        importancem = importance['importances_mean']
        importancev = importance['importances_std']**2
        sorted_ind = np.flip(np.argsort(importancem))
        vars = ['Adv','Adiab','Diab']
        dom = vars[sorted_ind[0]]
        mapping = {'Adv':1/6,'Adiab':3/6,'Diab':5/6}
        return mapping.get(dom,np.nan), importancem[sorted_ind[0]], importancem[sorted_ind[1]], importancem[sorted_ind[2]],  importancev[sorted_ind[0]], importancev[sorted_ind[1]], importancev[sorted_ind[2]]

def plot_orders(data,pos,draw_label=True):
    ax = fig.add_subplot(pos,projection=crs)
    ax.coastlines(resolution='110m',color='black')
    ax.contourf(lon_vals, lat_vals, data.values,cmap=cmap_order,levels=[0,1/3,2/3,1],antialiased=True,transform=trans)
    ax.gridlines(draw_labels=draw_label,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    t=ax.text(0.025,0.975,'(a) Most importance feature', ha='left',va='top', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))
    ax.set_facecolor('grey')

def plot_mags(data,pos,draw_label=True,n=1):
    ax = fig.add_subplot(pos,projection=crs)
    ax.coastlines(resolution='110m',color='black')
    ax.contourf(lon_vals, lat_vals, data.values,cmap=cmap_mags,antialiased=True,transform=trans) #,norm=norm_mags)
    ax.gridlines(draw_labels=draw_label,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    t=ax.text(0.025,0.975,f'{n} R2', ha='left',va='top', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))

def plot_vars(data,pos,draw_label=True,n=1):
    ax = fig.add_subplot(pos,projection=crs)
    ax.coastlines(resolution='110m',color='black')
    ax.contourf(lon_vals, lat_vals, data.values,cmap=cmap_vars,antialiased=True,transform=trans) #,norm=norm_vars)
    ax.gridlines(draw_labels=draw_label,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    t=ax.text(0.025,0.975,f'{n} R2', ha='left',va='top', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))

if __name__ == "__main__":                                                                                            
    
    # Set up a local cluster
    start = time.time()

    cluster = LocalCluster(n_workers=12)
    client = Client(cluster)

    print(f'Cluster setup: {time.time()-start:.4f} sec')
    start = time.time()

    (order, first, second, third, vfirst, vsecond, vthird) = xr.apply_ufunc(permutation_masked,xr_in['T_anom'],xr_in['adv'],xr_in['adiab'],xr_in['diab'],
               input_core_dims=[['years'],['years'],['years'],['years']],
               output_core_dims=[[],[],[],[],[],[],[]],
               vectorize=True,
               dask='parallelized', # since func converts to numpy.array
               output_dtypes=['float','float','float','float','float','float','float'])

    order = order.compute()
    first = first.compute()
    second = second.compute()
    third = third.compute()
    vfirst = vfirst.compute()
    vsecond = vsecond.compute()
    vthird = vthird.compute()

    print(f'Permute computation: {(time.time()-start)/60:.4f} min')
    start = time.time()

    lat_vals = order.lat.values
    lon_vals = order.lon.values

    crs = ccrs.Robinson(0)
    trans = ccrs.PlateCarree(0)

    t=1
    colors = colormaps['Paired']
    cmap_order = ListedColormap([colors(2),colors(0),colors(6)])

    colors = plt.cm.Spectral_r(np.linspace(0, 1, 10))
    cmap_mags = ListedColormap(colors)
    norm_mags = Normalize(0,1)

    colors = plt.cm.pink(np.linspace(0, 1, 10))
    cmap_vars = ListedColormap(colors)
    norm_vars = Normalize()

    labels = np.array(['Adv','Adiab','Diab'])

    fig = plt.figure(constrained_layout=True,figsize=(10,8))
    gs = GridSpec(4,3, wspace=0.05, hspace=0.05, figure=fig,height_ratios=[0.8, 0.8, 1, 1])

    plot_orders(order,gs[0:2,:])
    cbar = fig.add_axes([0.9,0.6,0.025,0.3])
    plt.colorbar(cm.ScalarMappable(cmap=cmap_order),cax=cbar,
             format=FixedFormatter(labels),ticks=[1/6,3/6,5/6],
             label = None, orientation='vertical',ticklocation='right')

    plot_mags(first,gs[2,0],True,'(b) 1st M')
    plot_mags(second,gs[2,1],True,'(c) 2nd M')
    plot_mags(third,gs[2,2],True,'(d) 3rd M')
    cbar = fig.add_axes([1.05,0.3,0.025,0.2])
    plt.colorbar(cm.ScalarMappable(cmap=cmap_mags),cax=cbar,#norm=norm_mags,
             label = 'Average R2', orientation='vertical')
    
    plot_vars(vfirst,gs[3,0],True,'(b) 1st V')
    plot_vars(vsecond,gs[3,1],True,'(c) 2nd V')
    plot_vars(vthird,gs[3,2],True,'(d) 3rd V')
    cbar = fig.add_axes([1.05,0.05,0.025,0.2])
    plt.colorbar(cm.ScalarMappable(cmap=cmap_vars),cax=cbar,#norm=norm_vars,
             label = 'Variance R2', orientation='vertical')
    
    plt.savefig(outdir+f'permutation_domiance', bbox_inches='tight')
    plt.close()

    print(f'Plotting: {(time.time()-start)/60:.4f} min')
