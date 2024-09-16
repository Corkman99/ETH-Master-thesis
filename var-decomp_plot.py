"""
Variance Decomposition Plot
"""

import xarray as xr
import numpy as np
import pandas as pd
import xskillscore as xs

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, TwoSlopeNorm, SymLogNorm, LogNorm, PowerNorm, Normalize, CenteredNorm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import iqr

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

"""
indir_comp = 'data/covariance/components.nc'
indir_t = 'data/covariance/t_anom.nc'
outdir = 'plots/var_decomp/'

# data load
xr_comp = xr.open_dataset(indir_comp)
xr_t = xr.open_dataset(indir_t)

vars = ['adv','adiab','diab'] 
for var in vars:
    xr_comp[var] = xr_comp[var].where(xr_comp[var] <= 128,np.nan)
vars=['adv_adiab','adv_diab','diab_adiab']
for var in vars:
    xr_comp[var] = xr_comp[var].where((xr_comp[var] <= 128) & (xr_comp[var] >= -128),np.nan)

bounds = np.arange(0,128,8)
bounds =[-128,-64,-32,-16,-8,-4,-2,-1,0,1,2,4,8,16,32,64,128]
bounds= range(9)
colors = plt.cm.Spectral(np.flip(np.linspace(0, 0.5, len(bounds)-1)))  # Using Viridis colormap for the range 0-10
cmap1 = ListedColormap(colors)

bounds = np.arange(-128,128,4)
bounds = [0,1,2,4,8,16,32,64,128]
bounds = range(17)
colors = plt.cm.Spectral(np.flip(np.linspace(0, 1, len(bounds)-1)))  # Using Viridis colormap for the range 0-10
cmap2 = ListedColormap(colors)

crs = ccrs.PlateCarree(0)

# extract lon/lat values
lat_vals = xr_comp.lat.values
lon_vals = xr_comp.lon.values

# labels/titles
labels = ['(a) Var(adv)','(b) Var(adiab)','(c) Var(diab)', 
          '(d) Cov(adv,adiab)','(e) Cov(adv,diab)','(f) Cov(diab,adiab)']
title_string = "Variance / Covariance Decomposition of TX1day"
filename = 'var_decomp'

rows=2
cols=3

# Plot
# define plot function for each
def do_plot(var,pos,label,ext='both',cmap='PiYG',norm=Normalize(),levels=0,
            draw_labels=[]):
    ax = fig.add_subplot(pos,projection=crs)
    ax.coastlines(resolution='110m',color='black')
    ax.contourf(lon_vals,lat_vals,xr_comp[var].values,transform=crs,
                extend=ext,cmap=cmap,norm=norm,levels=levels)
    t=ax.text(0.975, 0.05,label, ha='right',va='bottom', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))
    gl = ax.gridlines(crs=crs, draw_labels=draw_labels,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    ax.set_facecolor("grey") 

    XTEXT_SIZE = 6
    YTEXT_SIZE = 6
    gl.xlabel_style = {'size': XTEXT_SIZE}
    gl.ylabel_style = {'size':YTEXT_SIZE}

# create figure
fig = plt.figure(constrained_layout=True,figsize=(16,7))
gs = GridSpec(rows, cols, wspace=0.05, hspace=0.05, figure=fig)

norm = SymLogNorm(1,linscale=0.2,vmin=0,vmax=128,base=2)
levels=[0,1,2,4,8,16,32,64,128]
do_plot('adv',gs[0,0],'(a) Var(adv)',ext='max',cmap=cmap1,norm=norm,levels=levels,draw_labels={'left':'y','top':'x'})
do_plot('adiab',gs[0,1],'(b) Var(adiab)',ext='max',cmap=cmap1,norm=norm,levels=levels,draw_labels={'top':'x'})
do_plot('diab',gs[0,2],'(c) Var(diab)',ext='max',cmap=cmap1,norm=norm,levels=levels,draw_labels={'right':'y','top':'x'})
# cbar_ax1 = fig.add_axes([1.02, 0.55, 0.02, 0.35])
#cbar_ax1 = fig.add_axes([0.1, 1.02, 0.8, 0.025])
#cbar=plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap1),cax=cbar_ax1,ticks=levels,
                  #label = 'Variance', orientation='horizontal')
norm = SymLogNorm(1,linscale=0.2,vmin=-128,vmax=128,base=2)
levels=[-128,-64,-32,-16,-8,-4,-2,-1,0,1,2,4,8,16,32,64,128]
do_plot('adv_adiab',gs[1,0],'(d) Cov(adv,adiab)',cmap=cmap2,norm=norm,levels=levels,draw_labels={'left':'y','bottom':'x'})
do_plot('adv_diab',gs[1,1],'(e) Cov(adv,diab)',cmap=cmap2,norm=norm,levels=levels,draw_labels={'bottom':'x'})
do_plot('diab_adiab',gs[1,2],'(f) Cov(diab,adiab)',cmap=cmap2,norm=norm,levels=levels,draw_labels={'right':'y','bottom':'x'})

cbar_ax2 = fig.add_axes([0.1,0.475,0.8,0.025])
cbar=plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap2), boundaries=levels,cax=cbar_ax2,
                  label = None, orientation='horizontal',ticks=levels)
fig.suptitle(title_string, fontsize=15)

plt.savefig(outdir+filename, bbox_inches='tight')
plt.close()

"""
"""
#Variance plot:
title_string = "Variance of TX1day"
filename = 'var_tanom'

#xr_t['var'] = xr_t['var'].where(xr_t['var'] <= 16,np.nan)

# create figure
fig = plt.figure(constrained_layout=True,figsize=(15,6))

bounds = np.arange(-128,128,4)
bounds = [0,1,2,4,8,16,32,64,128]
bounds = range(17)
colors = plt.cm.Spectral(np.flip(np.linspace(0, 1, len(bounds)-1)))  # Using Viridis colormap for the range 0-10
cmap3 = ListedColormap(colors)

ax = fig.add_subplot(projection=crs)
ax.coastlines(resolution='110m',color='black')
cs = ax.contourf(lon_vals,lat_vals,xr_t['var'].values,transform=crs,
            extend='neither',cmap=cmap3,norm=norm,levels=[0,1,2,4,8,16,32,64,128])
ax.gridlines(crs=crs, draw_labels={'top':'x','bottom':'x','left':'y'},linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.set_facecolor("grey") 

plt.colorbar(cs,label = None, orientation='vertical',ticks=[0,1,2,4,8,16,32,64,128])
plt.suptitle(title_string, fontsize=15)

plt.savefig(outdir+filename, bbox_inches='tight')
plt.close()


# Plot for ordering contributions
from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib import colormaps

def find_order(var1,var2,var3,advadiab,advdiab,diabadiab):
    ret = np.argsort([var1,var2,var3,advadiab,advdiab,diabadiab])
    return (2*ret[5]+1)/12, (2*ret[4]+1)/12, (2*ret[3]+1)/12, (2*ret[2]+1)/12, (2*ret[1]+1)/12, (2*ret[0]+1)/12

xr_comp = xr.open_dataset('data/covariance/components.nc')
orders = xr.apply_ufunc(find_order,
                        xr_comp['adv'],xr_comp['adiab'],xr_comp['diab'],
                        xr_comp['adv_adiab'],xr_comp['adv_diab'],xr_comp['diab_adiab'],
                        input_core_dims=[[],[],[],[],[],[]],
                        output_core_dims=[[],[],[],[],[],[]],
                        vectorize=True,
                        dask='parallelized', # since func converts to numpy.array
                        output_dtypes=['float64','float64','float64','float64','float64','float64'])

df_order = xr.Dataset({'one':orders[0],'two':orders[1],'three':orders[2],'four':orders[3],'five':orders[4],'six':orders[5]}).compute()

lat_vals = df_order.lat.values
lon_vals = df_order.lon.values

t=1
colors = colormaps['tab20']
cmap = ListedColormap([colors(16),colors(2),colors(6),colors(12),colors(8),colors(0)])
labels = np.array(['Var(Adv)','Var(Adiab)','Var(Diab)','Cov(Adv,Adiab)','Cov(Adv,Diab)','Cov(Diab,Adiab)'])
def plot_em(var,pos,label,draw_label):
    ax = fig.add_subplot(pos,projection=ccrs.PlateCarree(0))
    ax.coastlines(resolution='110m',color='black')
    ax.contourf(lon_vals, lat_vals, df_order[var],cmap=cmap,levels=[0,1/6,2/6,3/6,4/6,5/6,1],antialiased=True)
    ax.gridlines(crs=ccrs.PlateCarree(0), draw_labels=draw_label,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    t=ax.text(0.975, 0.05,label, ha='right',va='bottom', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))

fig = plt.figure(constrained_layout=True,figsize=(16,7))
gs = GridSpec(2, 3, wspace=0.05, hspace=0.05, figure=fig)
plot_em('one',gs[0,0],'(a) Largest',draw_label={'left':'y','top':'x'})
plot_em('two',gs[0,1],'(b) 2nd Largest',draw_label={'top':'x'})
plot_em('three',gs[0,2],'(c) 3rd Largest',draw_label={'right':'y','top':'x'})
plot_em('four',gs[1,0],'(d) 3rd Smallest',draw_label={'left':'y','bottom':'x'})
plot_em('five',gs[1,1],'(e) 2nd Smallest',draw_label={'bottom':'x'})
plot_em('six',gs[1,2],'(f) Smallest',draw_label={'right':'y','bottom':'x'})

cbar = fig.add_axes([0.1,0.47,0.8,0.025])
plt.colorbar(cm.ScalarMappable(cmap=cmap),cax=cbar,
             format=FixedFormatter(labels),ticks=[1/12,3/12,5/12,7/12,9/12,11/12],
             label = None, orientation='horizontal',ticklocation='top')
fig.suptitle('Ordering of Variance / Covariance Decomposition Terms', fontsize=15)

plt.savefig(outdir+'var_decomp_ordered', bbox_inches='tight')
plt.close()

"""

"""
# dominance plot
outdir = 'plots/final/'

def dominant_mean(var1,var2,var3):
    names = ['Adv','Adiab','Diab']
    vars = [var1,var2,var3]
    ret = np.flip(np.argsort(vars))
    if vars[ret[0]] >= 2*vars[ret[1]]:
        dom = names[ret[0]]
    else:
        if vars[ret[0]] > 0 and vars[ret[1]] > 0 and vars[ret[2]] < 0:
            dom = names[ret[0]] + '/' + names[ret[1]]
        else:
            dom = 'all three'
    mapping = {'Adv':1/14,'Adiab':3/14,'Diab':5/14,
               'Adv/Adiab':7/14,'Adiab/Adv':7/14,
               'Adv/Diab':9/14,'Diab/Adv':9/14,
               'Adiab/Diab':11/14,'Diab/Adiab':11/14,
               'all three':13/14}
    
    return mapping.get(dom,np.nan)

def dominant_var(var1,var2,var3):
    names = ['Adv','Adiab','Diab']
    vars = [var1,var2,var3]
    ret = np.flip(np.argsort(vars))
    if vars[ret[0]] >= 2*vars[ret[1]]:
        dom = names[ret[0]]
    else:
        if vars[ret[1]] >= 2*vars[ret[2]]:
            dom = names[ret[0]] + '/' + names[ret[1]]
        else:
            dom = 'all three'
    mapping = {'Adv':1/14,'Adiab':3/14,'Diab':5/14,
               'Adv/Adiab':7/14,'Adiab/Adv':7/14,
               'Adv/Diab':9/14,'Diab/Adv':9/14,
               'Adiab/Diab':11/14,'Diab/Adiab':11/14,
               'all three':13/14}
    
    return mapping.get(dom,np.nan)

# mean
indir = '/net/litho/atmosdyn/roethlim/data/lastvar/era5/upload/TX1day/data/figures/'
infile = 'TX1day_decomposition_era5_v10_final.nc'
xr_in = xr.open_dataset(indir + '/' + infile,
                        chunks = {'years' : 42,'lon': 242, 'lat': 182}).mean(dim='lev',skipna=True)
xr_in = xr_in.drop_sel(years = 1979)
mean = xr_in.mean(dim='years')

mean = xr.apply_ufunc(dominant_mean,
                        mean['adv'],mean['adiab'],mean['diab'],
                        input_core_dims=[[],[],[]],
                        output_core_dims=[[]],
                        vectorize=True,
                        dask='parallelized', # since func converts to numpy.array
                        output_dtypes=['float'])

# var
indir_comp = 'data/covariance/components.nc'
var = xr.open_dataset(indir_comp)

indir_t = 'data/covariance/t_anom.nc'
var_t = xr.open_dataset(indir_t)

var = xr.apply_ufunc(dominant_var, 
                     var['adv'],var['adiab'],var['diab'],
                     input_core_dims=[[],[],[]],
                     output_core_dims=[[]],
                     vectorize=True,
                     dask='parallelized', # since func converts to numpy.array
                     output_dtypes=['float'])

# plotting
from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib import colormaps

lat_vals = var.lat.values
lon_vals = var.lon.values

crs = ccrs.Robinson(0)
trans = ccrs.PlateCarree(0)

t=1
colors = colormaps['Paired']
cmap = ListedColormap([colors(2),colors(0),colors(6),colors(10),colors(5),colors(7),colors(8)])
labels = np.array(['Adv','Adiab','Diab','Adv/Adiab','Adv/Diab','Adiab/Diab','all three'])
def plot_em(data,pos,label,draw_label):
    ax = fig.add_subplot(pos,projection=crs)
    ax.coastlines(resolution='110m',color='black')
    ax.contourf(lon_vals, lat_vals, data.values,cmap=cmap,levels=[0,1/7,2/7,3/7,4/7,5/7,6/7,1],antialiased=True,transform=trans)
    ax.gridlines(draw_labels=draw_label,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    t=ax.text(0.025,0.975,label, ha='left',va='top', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))

fig = plt.figure(constrained_layout=True,figsize=(8,8))
gs = GridSpec(2,1, wspace=0.05, hspace=0.05, figure=fig)
plot_em(mean,gs[0,0],'(a) Mean',draw_label={'left':'y','top':'x','right':'y'})
plot_em(var,gs[1,0],'(b) Variance',draw_label={'left':'y','bottom':'x','right':'y'})

cbar = fig.add_axes([1.05,0.2,0.025,0.6])
plt.colorbar(cm.ScalarMappable(cmap=cmap),cax=cbar,
             format=FixedFormatter(labels),ticks=[1/14,3/14,5/14,7/14,9/14,11/14,13/14],
             label = None, orientation='vertical',ticklocation='right')
#fig.suptitle('Dominant Mean and Variance Components', fontsize=15)

plt.savefig(outdir+'dominant', bbox_inches='tight')
plt.close()

"""

"""
# Pure Largest

# plot for just largest var, and largest mean contributor (only looking at variances)
outdir = 'plots/var_decomp/'

def largest(var1,var2,var3):
    ord = ['Adv','Adiab','Diab']
    ret = np.flip(np.argsort([var1,var2,var3]))

    return (2*ret[0]+1)/6

# mean
indir = '/net/litho/atmosdyn/roethlim/data/lastvar/era5/upload/TX1day/data/figures/'
infile = 'TX1day_decomposition_era5_v10_final.nc'
xr_in = xr.open_dataset(indir + '/' + infile,
                        chunks = {'years' : 42,'lon': 242, 'lat': 182}).mean(dim='lev',skipna=True)
xr_in = xr_in.drop_sel(years = 1979)
mean = xr_in.mean(dim='years')

mean = xr.apply_ufunc(largest,
                        mean['adv'],mean['adiab'],mean['diab'],
                        input_core_dims=[[],[],[]],
                        output_core_dims=[[]],
                        vectorize=True,
                        dask='parallelized', # since func converts to numpy.array
                        output_dtypes=['float'])

# var
indir_comp = 'data/covariance/components.nc'
var = xr.open_dataset(indir_comp)

var = xr.apply_ufunc(largest, 
                     var['adv'],var['adiab'],var['diab'],
                     input_core_dims=[[],[],[]],
                     output_core_dims=[[]],
                     vectorize=True,
                     dask='parallelized', # since func converts to numpy.array
                     output_dtypes=['float'])

# plotting
from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib import colormaps

lat_vals = var.lat.values
lon_vals = var.lon.values

t=1
colors = colormaps['Set2']
cmap = ListedColormap([colors(4),colors(2),colors(6)]) #,colors(3),colors(0),colors(1),colors(7)])
labels = np.array(['Adv','Adiab','Diab']) #,'Adv/Adiab','Adv/Diab','Adiab/Diab','all three'])
def plot_em(data,pos,label,draw_label):
    ax = fig.add_subplot(pos,projection=ccrs.PlateCarree(0))
    ax.coastlines(resolution='110m',color='black')
    ax.contourf(lon_vals, lat_vals, data.values,cmap=cmap,levels=[0,1/3,2/3,1],antialiased=True)
    ax.gridlines(crs=ccrs.PlateCarree(0), draw_labels=draw_label,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    t=ax.text(0.975, 0.05,label, ha='right',va='bottom', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))

fig = plt.figure(constrained_layout=True,figsize=(8,8))
gs = GridSpec(2,1, wspace=0.05, hspace=0.05, figure=fig)
plot_em(mean,gs[0,0],'(a) Mean',draw_label={'left':'y','top':'x','right':'y'})
plot_em(var,gs[1,0],'(b) Variance',draw_label={'left':'y','bottom':'x','right':'y'})

cbar = fig.add_axes([1,0.2,0.025,0.6])
plt.colorbar(cm.ScalarMappable(cmap=cmap),cax=cbar,
             format=FixedFormatter(labels),ticks=[1/6,3/6,5/6],
             label = None, orientation='vertical',ticklocation='right')
fig.suptitle('Largest Mean and Variance Components', fontsize=15)

plt.savefig(outdir+'largest', bbox_inches='tight')
plt.close()

"""

# PLOTTING ALL TOGETHER:

indir_comp = 'data/covariance/components.nc'
indir_t = 'data/covariance/t_anom.nc'
outdir = 'plots/final/'

# data load
xr_comp = xr.open_dataset(indir_comp)
xr_t = xr.open_dataset(indir_t)

vars = ['adv','adiab','diab'] 
for var in vars:
    xr_comp[var] = xr_comp[var].where(xr_comp[var] <= 128,np.nan)
    #print(var + ': ' + str(xr.where(xr_comp[var] <= 128,0,1).sum()))
vars=['adv_adiab','adv_diab','diab_adiab']
for var in vars:
    xr_comp[var] = xr_comp[var].where((xr_comp[var] <= 128) & (xr_comp[var] >= -128),np.nan)
    #print(var + ': ' + str(xr.where((xr_comp[var] <= 128) & (xr_comp[var] >= -128),0,1).sum()))

bounds= range(9)
colors = plt.cm.Spectral(np.flip(np.linspace(0, 0.5, len(bounds)-1)))  # Using Viridis colormap for the range 0-10
cmap1 = ListedColormap(colors)

bounds = range(17)
colors = plt.cm.Spectral(np.flip(np.linspace(0, 1, len(bounds)-1)))  # Using Viridis colormap for the range 0-10
cmap2 = ListedColormap(colors)

#crs = trans = ccrs.PlateCarree(0)
crs = ccrs.Robinson(0)
trans = ccrs.PlateCarree(0)

# extract lon/lat values
lat_vals = xr_comp.lat.values
lon_vals = xr_comp.lon.values

# labels/titles
title_string = "Variance / Covariance Decomposition of TX1day"
filename = 'var_decomp_full_sin'

rows=4
cols=3

# Plot
# define plot function for each
def do_plot(var,pos,label,ext='both',cmap='PiYG',norm=Normalize(),levels=0,
            draw_labels={}):
    ax = fig.add_subplot(pos,projection=crs)
    ax.coastlines(resolution='110m',color='black')
    ax.contourf(lon_vals,lat_vals,xr_comp[var].values,transform=trans,
                extend=ext,cmap=cmap,norm=norm,levels=levels)
    t=ax.text(0.025, 1.1,label, ha='left',va='top', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))
    gl = ax.gridlines(draw_labels=draw_labels,linewidth=1, color='gray', alpha=0.5, linestyle='--') #crs=crs
    ax.set_facecolor("grey") 

    XTEXT_SIZE = 6
    YTEXT_SIZE = 6
    gl.xlabel_style = {'size': XTEXT_SIZE}
    gl.ylabel_style = {'size':YTEXT_SIZE}
    print('plotted')

norm = SymLogNorm(1,linscale=0.2,vmin=0,vmax=128,base=2)

# create figure
fig = plt.figure(constrained_layout=True,figsize=(14,10))
gs = GridSpec(rows, cols, wspace=0.05, hspace=0.05, figure=fig,height_ratios=[0.8, 0.8, 1, 1])

ax = fig.add_subplot(gs[0:2,:],projection=crs)
ax.coastlines(resolution='110m',color='black')
ax.contourf(lon_vals,lat_vals,xr_t['var'].values,transform=trans,
            extend='neither',cmap=cmap1,norm=norm,levels=[0,1,2,4,8,16,32,64,128])
t=ax.text(0.025, 1.025,"(a) Var(T_anom)", ha='left',va='top', transform=ax.transAxes)
t.set_bbox(dict(facecolor='white', edgecolor='black'))
ax.gridlines(draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.set_facecolor("grey") 

levels=[0,1,2,4,8,16,32,64,128]
do_plot('adv',gs[2,0],'(b) Var(adv)',ext='max',cmap=cmap1,norm=norm,levels=levels) #,draw_labels={'left':'y','top':'x'})
do_plot('adiab',gs[2,1],'(c) Var(adiab)',ext='max',cmap=cmap1,norm=norm,levels=levels) #,draw_labels={'top':'x'})
do_plot('diab',gs[2,2],'(d) Var(diab)',ext='max',cmap=cmap1,norm=norm,levels=levels) #,draw_labels={'right':'y','top':'x'})

norm = SymLogNorm(1,linscale=0.2,vmin=-128,vmax=128,base=2)
levels=[-128,-64,-32,-16,-8,-4,-2,-1,0,1,2,4,8,16,32,64,128]
do_plot('adv_adiab',gs[3,0],'(e) Cov(adv,adiab)',cmap=cmap2,norm=norm,levels=levels) #,draw_labels={'left':'y','bottom':'x'})
do_plot('adv_diab',gs[3,1],'(f) Cov(adv,diab)',cmap=cmap2,norm=norm,levels=levels) #,draw_labels={'bottom':'x'})
do_plot('diab_adiab',gs[3,2],'(g) Cov(diab,adiab)',cmap=cmap2,norm=norm,levels=levels) #,draw_labels={'right':'y','bottom':'x'})

cbar_ax2 = fig.add_axes([0.1,-0.05,0.8,0.025])
cbar=plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap2), boundaries=levels,cax=cbar_ax2,
                  label = r'$^\circ$C$^2$', orientation='horizontal',ticks=levels)
#fig.suptitle(title_string, fontsize=15)

plt.savefig(outdir+filename, bbox_inches='tight')
plt.close()


"""
# PLOTTING ALL TOGETHER: next decomposition

indir_comp = 'data/covariance/components.nc'
indir_t = 'data/covariance/t_anom.nc'
outdir = 'plots/final/'

# data load
xr_comp = xr.open_dataset(indir_comp)
xr_t = xr.open_dataset(indir_t)

xr_comp['adv_adiab2'] = xr_comp['adv'] + xr_comp['adiab'] + xr_comp['adv_adiab'] - xr_comp['diab']
xr_comp['adv_diab2'] = xr_comp['adv'] + xr_comp['diab'] + xr_comp['adv_diab'] - xr_comp['adiab']
xr_comp['diab_adiab2'] = xr_comp['diab'] + xr_comp['adiab'] + xr_comp['diab_adiab'] - xr_comp['adv']

vars = ['adv','adiab','diab'] 
for var in vars:
    xr_comp[var] = xr_comp[var].where(xr_comp[var] <= 128,np.nan)
    #print(var + ': ' + str(xr.where(xr_comp[var] <= 128,0,1).sum()))
vars=['adv_adiab2','adv_diab2','diab_adiab2']
for var in vars:
    xr_comp[var] = xr_comp[var].where((xr_comp[var] <= 128) & (xr_comp[var] >= -128),np.nan)
    #print(var + ': ' + str(xr.where((xr_comp[var] <= 128) & (xr_comp[var] >= -128),0,1).sum()))

bounds= range(9)
colors = plt.cm.Spectral(np.flip(np.linspace(0, 0.5, len(bounds)-1)))  # Using Viridis colormap for the range 0-10
cmap1 = ListedColormap(colors)

bounds = range(17)
colors = plt.cm.Spectral(np.flip(np.linspace(0, 1, len(bounds)-1)))  # Using Viridis colormap for the range 0-10
cmap2 = ListedColormap(colors)

#crs = trans = ccrs.PlateCarree(0)
crs = ccrs.Robinson(0)
trans = ccrs.PlateCarree(0)

# extract lon/lat values
lat_vals = xr_comp.lat.values
lon_vals = xr_comp.lon.values

# labels
title_string = "Variance / Covariance Decomposition of TX1day"
filename = 'var_decomp2_full'

rows=4
cols=3

# Plot
# define plot function for each
def do_plot(var,pos,label,ext='both',cmap='PiYG',norm=Normalize(),levels=0,
            draw_labels={}):
    ax = fig.add_subplot(pos,projection=crs)
    ax.coastlines(resolution='110m',color='black')
    ax.contourf(lon_vals,lat_vals,xr_comp[var].values,transform=trans,
                extend=ext,cmap=cmap,norm=norm,levels=levels)
    t=ax.text(0.025, 1.1,label, ha='left',va='top', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))
    gl = ax.gridlines(draw_labels=draw_labels,linewidth=1, color='gray', alpha=0.5, linestyle='--') #crs=crs
    XTEXT_SIZE = 8
    YTEXT_SIZE = 8
    gl.xlabel_style = {'size': XTEXT_SIZE}
    gl.ylabel_style = {'size':YTEXT_SIZE}
    ax.set_facecolor("grey") 
    print('plotted')

norm = SymLogNorm(1,linscale=0.2,vmin=0,vmax=128,base=2)

# create figure
fig = plt.figure(constrained_layout=True,figsize=(14,10))
gs = GridSpec(rows, cols, wspace=0.05, hspace=0.05, figure=fig,height_ratios=[0.8, 0.8, 1, 1])

ax = fig.add_subplot(gs[0:2,:],projection=crs)
ax.coastlines(resolution='110m',color='black')
ax.contourf(lon_vals,lat_vals,xr_t['var'].values,transform=trans,
            extend='neither',cmap=cmap1,norm=norm,levels=[0,1,2,4,8,16,32,64,128])
t=ax.text(0.025, 1.025,"(a) Var(T_anom)", ha='left',va='top', transform=ax.transAxes)
t.set_bbox(dict(facecolor='white', edgecolor='black'))
ax.gridlines(draw_labels=True,linewidth=1, color='gray', alpha=0.5, linestyle='--')
ax.set_facecolor("grey") 

levels=[0,1,2,4,8,16,32,64,128]
do_plot('adv',gs[2,0],'(b) Var(adv)',ext='max',cmap=cmap1,norm=norm,levels=levels) #,draw_labels={'left':'y','top':'x'})
do_plot('adiab',gs[2,1],'(c) Var(adiab)',ext='max',cmap=cmap1,norm=norm,levels=levels) #,draw_labels={'top':'x'})
do_plot('diab',gs[2,2],'(d) Var(diab)',ext='max',cmap=cmap1,norm=norm,levels=levels) #,draw_labels={'right':'y','top':'x'})

norm = SymLogNorm(1,linscale=0.2,vmin=-128,vmax=128,base=2)
levels=[-128,-64,-32,-16,-8,-4,-2,-1,0,1,2,4,8,16,32,64,128]
do_plot('adv_adiab2',gs[3,0],'(e) Var(adv+adiab)-Var(diab)',cmap=cmap2,norm=norm,levels=levels) #,draw_labels={'left':'y','bottom':'x'})
do_plot('adv_diab2',gs[3,1],'(f) Var(adv+diab)-Var(adiab)',cmap=cmap2,norm=norm,levels=levels) #,draw_labels={'bottom':'x'})
do_plot('diab_adiab2',gs[3,2],'(g) Var(diab+adiab)-Var(adv)',cmap=cmap2,norm=norm,levels=levels) #,draw_labels={'right':'y','bottom':'x'})

cbar_ax2 = fig.add_axes([0.1,-0.05,0.8,0.025])
cbar=plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap2), boundaries=levels,cax=cbar_ax2,
                  label = None, orientation='horizontal',ticks=levels)
#fig.suptitle(title_string, fontsize=15)

plt.savefig(outdir+filename, bbox_inches='tight')
plt.close()
"""