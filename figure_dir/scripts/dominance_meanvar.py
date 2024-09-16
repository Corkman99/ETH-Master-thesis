
import xarray as xr
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

from matplotlib.ticker import FixedFormatter, FixedLocator

import dask 

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Configuration
n_cpus = 16  # Number of CPUs to use for parallelization
dask.config.set(scheduler='threads', num_workers=n_cpus)

# Paths
indir = '/net/litho/atmosdyn/roethlim/data/lastvar/era5/upload/TX1day/data/figures/'
infile = 'TX1day_decomposition_era5_v10_final.nc'
outdir = '/home/mfroelich/Thesis/figure_dir/plots/'

mapping = {'Adv':13/14,'Adv/Adiab':11/14,'Adiab/Adv':11/14,
           'Adiab':9/14,'Adiab/Diab':7/14,'Diab/Adiab':7/14,
           'Diab':5/14,'Adv/Diab':3/14,'Diab/Adv':3/14,
           'all three':1/14}

if __name__ == "__main__":
    
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
        
        return mapping.get(dom,np.nan)

    # MEAN
    xr_in = xr.open_dataset(indir + '/' + infile,chunks = {'years' : 42,'lon': 242, 'lat': 182}).mean(dim='lev',skipna=True)
    xr_in = xr_in.drop_sel(years = 1979)
    mean = xr_in.mean(dim='years')

    mean = xr.apply_ufunc(dominant_mean,
                            mean['adv'],mean['adiab'],mean['diab'],
                            input_core_dims=[[],[],[]],
                            output_core_dims=[[]],
                            vectorize=True,
                            dask='parallelized', # since func converts to numpy.array
                            output_dtypes=['float'])

    # VAR
    indir_comp = '/home/mfroelich/Thesis/data/covariance/components.nc'
    var = xr.open_dataset(indir_comp)

    indir_t = '/home/mfroelich/Thesis/data/covariance/t_anom.nc'
    var_t = xr.open_dataset(indir_t)

    var = xr.apply_ufunc(dominant_var, 
                        var['adv'],var['adiab'],var['diab'],
                        input_core_dims=[[],[],[]],
                        output_core_dims=[[]],
                        vectorize=True,
                        dask='parallelized', # since func converts to numpy.array
                        output_dtypes=['float'])

    lat_vals = var.lat.values
    lon_vals = var.lon.values

    crs = ccrs.Robinson(0)
    trans = ccrs.PlateCarree(0)

    cmap = ListedColormap(list(reversed(['#D1BA48','#D19F62','#D36159','#D5AFF0','#5994D3','#66D66E','#F0F2F1'])))
    labels = np.flip(np.array(['Adv','Adv/Adiab','Adiab','Adiab/Diab','Diab','Adv/Diab','All']))
    def plot_em(data,pos,label,draw_label):
        ax = fig.add_subplot(pos,projection=crs)
        ax.coastlines(resolution='110m',color='black')
        ax.contourf(lon_vals, lat_vals, data.values,cmap=cmap,levels=[0,1/7,2/7,3/7,4/7,5/7,6/7,1],antialiased=True,transform=trans)
        gl = ax.gridlines(draw_labels=draw_label,linewidth=1, color='gray', alpha=0.5, linestyle='--')
        t=ax.text(0.025,0.975,label, ha='left',va='top', transform=ax.transAxes)
        t.set_bbox(dict(facecolor='white', edgecolor='black'))

        XTEXT_SIZE = 8
        YTEXT_SIZE = 8
        gl.xlabel_style = {'size': XTEXT_SIZE}
        gl.ylabel_style = {'size':YTEXT_SIZE}

    fig = plt.figure(constrained_layout=True,figsize=(8,8))
    gs = GridSpec(2,1, wspace=0.05, hspace=0.05, figure=fig)
    plot_em(mean,gs[0,0],'(a) Mean',draw_label={'left':'y','top':'x','right':'y'})
    plot_em(var,gs[1,0],'(b) Variance',draw_label={'left':'y','bottom':'x','right':'y'})

    cbar = fig.add_axes([1.05,0.2,0.025,0.6])
    plt.colorbar(cm.ScalarMappable(cmap=cmap),cax=cbar,
                format=FixedFormatter(labels),ticks=[1/14,3/14,5/14,7/14,9/14,11/14,13/14],
                label = None, orientation='vertical',ticklocation='right')

    plt.savefig(f'{outdir}dominant_meanvar.png', bbox_inches='tight')
    plt.close()