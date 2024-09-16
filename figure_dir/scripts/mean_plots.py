"""
Plots of Mean-behvaiour

"""
import numpy as np
import xarray as xr

import time

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, Normalize, BoundaryNorm
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

## File load ect
indir = '/net/litho/atmosdyn/roethlim/data/lastvar/era5/upload/TX1day/data/figures/'
infile = 'TX1day_decomposition_era5_v10_final.nc'
outdir = '/home/mfroelich/Thesis/figure_dir/plots/'

def plot_mags(data,pos,draw_label=True,n=1,norm=1,cmap=1,levels=0):
    ax = fig.add_subplot(pos,projection=crs)
    ax.coastlines(resolution='110m',color='black')
    ax.contourf(lon_vals, lat_vals, data.values,
                levels=levels,cmap=cmap,transform=trans,extend='both')
    gl = ax.gridlines(draw_labels=draw_label,ylocs=range(-60,61,30),
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    t=ax.text(0.01,0.99,n,ha='left',va='top', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))

    XTEXT_SIZE = 8
    YTEXT_SIZE = 8
    gl.xlabel_style = {'size': XTEXT_SIZE}
    gl.ylabel_style = {'size':YTEXT_SIZE}

if __name__ == "__main__":   

    xr_in = xr.open_dataset(indir + '/' + infile,chunks = {'years' : 42,'lon': 181, 'lat': 91}).mean(dim=['lev'],skipna=True)[['T_anom','adv','adiab','diab']]
    xr_in = xr_in.drop_sel(years = 1979)
    xr_in = xr_in.mean(dim=['years'])

    #vars = ['T_anom','adv','adiab','diab'] 
    #for var in vars:
    #    xr_in[var] = xr_in[var].where(xr_in[var] <= 15,np.nan)
    #    xr_in[var] = xr_in[var].where(xr_in[var] >= -15,np.nan)

    lat_vals = xr_in.lat.values
    lon_vals = xr_in.lon.values

    crs = ccrs.Robinson(0)
    trans = ccrs.PlateCarree(0)

    bounds = [-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14]
    colors = plt.cm.Spectral(np.flip(np.linspace(0, 1, len(bounds))))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds,ncolors=cmap.N,clip=False)

    fig = plt.figure(figsize=(15,8))
    gs = GridSpec(2,2,figure=fig,hspace=0.1,wspace=0.1)

    #plot_mags(xr_in['T_anom'],gs[0,0],{'left':True,'right':False,'top':True,'bottom':False},'(a) T\'',norm=norm,cmap=cmap,levels=bounds)
    #plot_mags(xr_in['adv'],gs[0,1],{'left':False,'right':True,'top':True,'bottom':False},'(b) adv T\'',norm=norm,cmap=cmap,levels=bounds)
    #plot_mags(xr_in['adiab'],gs[1,0],{'left':True,'right':False,'top':False,'bottom':True},'(c) adiab T\'',norm=norm,cmap=cmap,levels=bounds)
    #plot_mags(xr_in['diab'],gs[1,1],{'left':False,'right':True,'top':False,'bottom':True},'(d) diab T\'',norm=norm,cmap=cmap,levels=bounds)

    plot_mags(xr_in['T_anom'],gs[0,0],{'left':True,'right':False,'top':True,'bottom':False},'(a)',norm=norm,cmap=cmap,levels=bounds)
    plot_mags(xr_in['adv'],gs[0,1],{'left':False,'right':True,'top':True,'bottom':False},'(b)',norm=norm,cmap=cmap,levels=bounds)
    plot_mags(xr_in['adiab'],gs[1,0],{'left':True,'right':False,'top':False,'bottom':True},'(c)',norm=norm,cmap=cmap,levels=bounds)
    plot_mags(xr_in['diab'],gs[1,1],{'left':False,'right':True,'top':False,'bottom':True},'(d)',norm=norm,cmap=cmap,levels=bounds)

    cbar = fig.add_axes([1,0.2,0.02,0.6])
    clb = plt.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),cax=cbar,extend='both',
             label = None, orientation='vertical',boundaries = bounds)
    clb.ax.set_title(r'$^\circ$C')

    pos = cbar.get_position()
    new_pos = [pos.x0-0.05, pos.y0, pos.width, pos.height]
    cbar.set_position(new_pos)

    plt.tight_layout()
    plt.savefig(outdir+f'mean_contribution', bbox_inches='tight')
    plt.close()
