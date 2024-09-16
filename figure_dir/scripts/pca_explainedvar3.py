import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.ticker import FixedFormatter
from matplotlib.cm import ScalarMappable
from sklearn import preprocessing
import cartopy.crs as ccrs
import xarray as xr
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
import dask

# Configuration
#n_cpus = 20  # Number of CPUs to use for parallelization
#dask.config.set(scheduler='threads', num_workers=n_cpus)

# Paths
indir = '/net/litho/atmosdyn/roethlim/data/lastvar/era5/upload/TX1day/data/figures/'
infile = 'TX1day_decomposition_era5_v10_final.nc'
outdir = '/home/mfroelich/Thesis/figure_dir/plots/'

# Functions:
def get_dominant(mags):
    mags = [np.absolute(x) for x in mags]
    vars = ['Adv','Adiab','Diab']
    order = np.flip(np.argsort(mags))
    if mags[order[0]] >= np.sqrt(3)/2: # corners
        dom = vars[order[0]]
    else:
        if sum(mags) >= np.sqrt(2)/2 + 1: # center
            dom = 'all three'
        else: 
            dom = vars[order[0]] + '/' + vars[order[1]] # rest is NOT smallest coordinate

        mapping = {'Adv':13/14,'Adv/Adiab':11/14,'Adiab/Adv':11/14,
           'Adiab':9/14,'Adiab/Diab':7/14,'Diab/Adiab':7/14,
           'Diab':5/14,'Adv/Diab':3/14,'Diab/Adv':3/14,
           'all three':1/14}
        return mapping.get(dom,np.nan)
    
    # PCA Analysis
def pca_var(dfadv, dfadiab, dfdiab):
    df = np.stack((dfadv, dfadiab, dfdiab), axis=1)
    df = preprocessing.StandardScaler().fit_transform(df)
    pca = PCA(n_components=3,svd_solver='full').fit(df)
    explained_var1 = pca.explained_variance_ratio_[0]
    explained_var2 = pca.explained_variance_ratio_[1]
    first = pca.components_[0,:] #ndarray of shape (n_components, n_features)
    second = pca.components_[1,:] #ndarray of shape (n_components, n_features)
    if explained_var1+explained_var2 >= 0.95:
        dom1 = get_dominant([first[0],first[1],first[2]])
        dom2 = get_dominant([second[0],second[1],second[2]])
    else: 
        dom1 = np.nan
        dom2 = np.nan    
    
    mask = 0.0
    if explained_var1 >= 0.80:
        mask = 1.0

    return explained_var1, explained_var1 + explained_var2, dom1, dom2, mask

def add_varplot(var, pos, label, draw, cmap, norm):
    ax = fig.add_subplot(pos, projection=crs)
    ax.coastlines(resolution='110m', color='black')
    ax.contourf(lon_vals, lat_vals, df[var].values, extend='min', cmap=cmap,norm=norm,transform=trans,antialiased=True)
    ax.text(0.025, 1, label, ha='left', va='top', transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black'))
    ax.gridlines(draw_labels=draw, linewidth=1, color='gray', alpha=0.5, linestyle='--')

def add_domplot(var,pos,label,draw_label, cmap, hatches):
    ax = fig.add_subplot(pos,projection=crs)
    ax.coastlines(resolution='110m',color='black')
    ax.contourf(lon_vals, lat_vals, df[var].values,cmap=cmap,levels=[0,1/7,2/7,3/7,4/7,5/7,6/7,1],antialiased=True,transform=trans)
    #if hatches:
    #    l = ax.contour(lon_vals, lat_vals, df['mask'].values ,3 , hatches=[None, '..'],  alpha=0.25,transform=trans,colors='none')
        #for c in l.collections:
        #    c.set_edgecolor("face")
        #    c.set_linewidth(0.000000000001)
    ax.gridlines(draw_labels=draw_label,linewidth=1, color='gray', alpha=0.5, linestyle='--')
    t=ax.text(0.025,0.975,label, ha='left',va='top', transform=ax.transAxes)
    t.set_bbox(dict(facecolor='white', edgecolor='black'))
    ax.set_facecolor("#EBEBEB")

if __name__ == "__main__":

    # Load dataset and pre-process
    xr_in = xr.open_dataset(f"{indir}/{infile}", chunks={'years': 42, 'lon': 361, 'lat': 181}).mean(dim='lev', skipna=True)
    xr_in = xr_in.drop_sel(years=1979)

    out = xr.apply_ufunc(pca_var, xr_in['adv'], xr_in['adiab'], xr_in['diab'],
                        input_core_dims=[['years'], ['years'], ['years']],
                        output_core_dims=[[], [],[],[],[]],
                        vectorize=True,
                        dask='parallelized',
                        output_dtypes=['float', 'float','float','float','float'])

    df = xr.Dataset({'one': out[0], 'oneandtwo': out[1], 'dom1': out[2], 'dom2': out[3], 'mask': out[4]})

    # Plotting
    crs = ccrs.Robinson()
    trans = ccrs.PlateCarree()
    lat_vals = xr_in.lat.values
    lon_vals = xr_in.lon.values
    
    fig = plt.figure(layout='constrained', figsize=(12,7))
    gs = GridSpec(2,2, wspace=0.05, hspace=0.05, figure=fig)

    ref_levels = np.arange(0.4,1.05,0.05)
    levels = [0.4,0.50,0.60,0.70,0.80,0.85,0.90,0.95,1]
    label_levels = [str(int(x*100)) for x in levels]
    colors = plt.cm.Spectral_r(np.linspace(0.1,1,len(levels)-1))
    cmap1 = ListedColormap([colors[0],colors[0],
                            colors[1],colors[1],
                            colors[2],colors[2],
                            colors[3],colors[3],
                            colors[4],colors[5],colors[6],colors[7]])
    #cmap1 = 'Spectral_r'
    norm1 = Normalize(0.4,1)

    df = df.compute()
    add_varplot('one', gs[0,0], '(a) EV(PC1)', {'left': 'y', 'top': 'x'},cmap1,norm1)
    add_varplot('oneandtwo', gs[0,1], '(b) EV(PC1 + PC2)', {'right': 'y', 'top': 'x'},cmap1,norm1)
    cbar_ax1 = fig.add_axes([1, 0.55, 0.02, 0.4])
    plt.colorbar(ScalarMappable(cmap=cmap1,norm=norm1), cax=cbar_ax1, extend='min',
                 label='% of explained variance',orientation='vertical', ticks=levels, spacing='proportional',
                 format = FixedFormatter(label_levels))
    
    cmap2 = ListedColormap(list(reversed(['#D1BA48','#D19F62','#D36159','#D5AFF0','#5994D3','#66D66E','#7A7A7A'])))
    labels = np.flip(np.array(['Adv','Adv/Adiab','Adiab','Adiab/Diab','Diab','Adv/Diab','All']))
    add_domplot('dom1',gs[1,0], '(c) Dominant PC1 score', {'left': 'y', 'bottom': 'x'}, cmap2,False)
    add_domplot('dom2',gs[1,1], '(d) Dominant PC2 score', {'right': 'y', 'bottom': 'x'}, cmap2,False)
    cbar_ax2 = fig.add_axes([1,0.05,0.02,0.4])
    plt.colorbar(ScalarMappable(cmap=cmap2),cax=cbar_ax2,
                format=FixedFormatter(labels),ticks=[1/14,3/14,5/14,7/14,9/14,11/14,13/14],
                label = None, orientation='vertical',ticklocation='right')
    
    plt.savefig(f"{outdir}pca_variance_and_dominance.png", bbox_inches='tight')
    plt.close()
