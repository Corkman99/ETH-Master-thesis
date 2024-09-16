import xarray as xr
import numpy as np
import pandas as pd
import xskillscore as xs

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, TwoSlopeNorm, SymLogNorm, LogNorm, PowerNorm, Normalize, CenteredNorm, BoundaryNorm, LinearSegmentedColormap

from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA 
from sklearn import preprocessing

import cartopy.crs as ccrs

indir = '/net/litho/atmosdyn/roethlim/data/lastvar/era5/upload/TX1day/data/figures/'
infile = 'TX1day_decomposition_era5_v10_final.nc'
outdir = '/home/mfroelich/Thesis/figure_dir/plots/'

xr_in = xr.open_dataset(indir + '/' + infile,
                        chunks = {'years' : 42,'lon': 242, 'lat': 182}).mean(dim='lev',skipna=True)
# drop first year because of NaN
xr_in = xr_in.drop_sel(years = 1979)

level = 80
def pca_score_masked_dominant(dfadv,dfadiab,dfdiab):
    df = np.stack((dfadv,dfadiab,dfdiab),axis=1)
    df = preprocessing.StandardScaler().fit_transform(df)
    pca = PCA(n_components=3)
    pca_res = pca.fit(df)
    explained_var = pca_res.explained_variance_ratio_[0]
    first = pca_res.components_[0,:] #ndarray of shape (n_components, n_features)
    if explained_var >= level/100:
        vars = ['Adv','Adiab','Diab']
        mags = [first[0],first[1],first[2]]
        order = np.flip(np.argsort(mags))
        if mags[order[0]] >= np.sqrt(3)/2: # corners
            dom = vars[order[0]]
        else:
            if sum(mags) >= np.sqrt(2)/2 + 1: # center
                dom = 'all three'
            else: 
                dom = vars[order[0]] + '/' + vars[order[1]] # rest is NOT smallest coordinate

        mapping = {'Adv':1/14,'Adiab':3/14,'Diab':5/14,
                   'Adv/Adiab':7/14,'Adiab/Adv':7/14,
                   'Adv/Diab':9/14,'Diab/Adv':9/14,
                   'Adiab/Diab':11/14,'Diab/Adiab':11/14,
                   'all three':13/14}
        return mapping.get(dom,np.nan)
    else:
        return np.nan
    
def pca_score_masked_dominant2(dfadv,dfadiab,dfdiab):
    df = np.stack((dfadv,dfadiab,dfdiab),axis=1)
    df = preprocessing.StandardScaler().fit_transform(df)
    pca = PCA(n_components=3)
    pca_res = pca.fit(df)
    explained_var = pca_res.explained_variance_ratio_[0]
    first = pca_res.components_[0,:] #ndarray of shape (n_components, n_features)
    if explained_var >= level/100:
        vars = ['Adv','Adiab','Diab']
        mags = [first[0],first[1],first[2]]
        order = np.flip(np.argsort(mags))
        if mags[order[1]]+mags[order[2]] <= 0.5: # corners
            dom = vars[order[0]]
        else:
            if sum(mags) >= 1.65: # center
                dom = 'all three'
            else: 
                dom = vars[order[0]] + '/' + vars[order[1]] # rest is NOT smallest coordinate

        mapping = {'Adv':1/14,'Adiab':3/14,'Diab':5/14,
                   'Adv/Adiab':7/14,'Adiab/Adv':7/14,
                   'Adv/Diab':9/14,'Diab/Adv':9/14,
                   'Adiab/Diab':11/14,'Diab/Adiab':11/14,
                   'all three':13/14}
        return mapping.get(dom,np.nan)
    else:
        return np.nan
    
df = xr.apply_ufunc(pca_score_masked_dominant2,xr_in['adv'],xr_in['adiab'],xr_in['diab'],
               input_core_dims=[['years'],['years'],['years']],
               output_core_dims=[[]],
               vectorize=True,
               dask='parallelized', # since func converts to numpy.array
               output_dtypes=['float']).compute()

from matplotlib.ticker import FixedFormatter, FixedLocator
from matplotlib import colormaps

lat_vals = df.lat.values
lon_vals = df.lon.values

crs = ccrs.Robinson(0)
trans = ccrs.PlateCarree(0)

t=1
colors = colormaps['Paired']
cmap = ListedColormap([colors(2),colors(0),colors(6),colors(10),colors(5),colors(7),colors(8)])
labels = np.array(['Adv','Adiab','Diab','Adv/Adiab','Adv/Diab','Adiab/Diab','all three'])
def plot_em(data,pos,label=True,draw_label=True):
    ax = fig.add_subplot(pos,projection=crs)
    ax.coastlines(resolution='110m',color='black')
    ax.contourf(lon_vals, lat_vals, data.values,cmap=cmap,levels=[0,1/7,2/7,3/7,4/7,5/7,6/7,1],antialiased=True,transform=trans)
    ax.gridlines(draw_labels=draw_label,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    #t=ax.text(0.025,0.975,label, ha='left',va='top', transform=ax.transAxes)
    #t.set_bbox(dict(facecolor='white', edgecolor='black'))

fig = plt.figure(constrained_layout=True,figsize=(14,7))
gs = GridSpec(1,1,figure=fig)
plot_em(df,gs[0,0],draw_label=True)

cbar = fig.add_axes([1.05,0.3,0.025,0.4])
plt.colorbar(cm.ScalarMappable(cmap=cmap),cax=cbar,
             format=FixedFormatter(labels),ticks=[1/14,3/14,5/14,7/14,9/14,11/14,13/14],
             label = None, orientation='vertical',ticklocation='right')

plt.savefig(outdir+f'pca_dominant', bbox_inches='tight')
plt.close()