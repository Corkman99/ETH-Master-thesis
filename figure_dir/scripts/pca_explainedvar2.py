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

if __name__ == "__main__":

    # Load dataset and pre-process
    xr_in = xr.open_dataset(f"{indir}/{infile}", chunks={'years': 42, 'lon': 361, 'lat': 181}).mean(dim='lev', skipna=True)
    xr_in = xr_in.drop_sel(years=1979)

    # Levels and colormap configuration
    levels = [0, 5, 15, 30, 50, 70, 85, 95, 100]
    levels = [0,5,10,20,30,40,50,60,70,80,90,95,100]
    colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725', '#fdae61', '#f46d43', '#d73027']
    colors = plt.cm.Spectral_r(np.linspace(0,1,len(levels)-1))
    
    # Use ListedColormap instead of LinearSegmentedColormap
    custom_cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=custom_cmap.N)

    # PCA Analysis
    def pca_var(dfadv, dfadiab, dfdiab):
        df = np.stack((dfadv, dfadiab, dfdiab), axis=1)
        df = preprocessing.StandardScaler().fit_transform(df)
        pca = PCA(n_components=3)
        pca_res = pca.fit(df)
        # colormap is exclusive on upper boundary, inclusive in lower. So if we take np.floor() should be fine. 50.01 -> 50 True, 49.99 - 49 True
        return pca_res.explained_variance_ratio_[0], pca_res.explained_variance_ratio_[1], pca_res.explained_variance_ratio_[2]

    out = xr.apply_ufunc(pca_var, xr_in['adv'], xr_in['adiab'], xr_in['diab'],
                        input_core_dims=[['years'], ['years'], ['years']],
                        output_core_dims=[[], [], []],
                        vectorize=True,
                        dask='parallelized',
                        output_dtypes=['float', 'float', 'float'])

    df = xr.Dataset({'one': out[0], 'two': out[1], 'three': out[2]}).compute()*100
    df['one and two'] = df['one'] + df['two']

    # Plotting
    crs = ccrs.Robinson()
    trans = ccrs.PlateCarree()
    lat_vals = xr_in.lat.values
    lon_vals = xr_in.lon.values

    def add_plot(var, pos, label, draw):
        val = df[var].values
        ax = fig.add_subplot(pos, projection=crs)
        ax.coastlines(resolution='110m', color='black')
        ax.contourf(lon_vals, lat_vals, val, extend='neither', cmap=custom_cmap,norm=norm,transform=trans)
        ax.text(0.025, 1.07, label, ha='left', va='top', transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black'))
        ax.gridlines(draw_labels=draw, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    fig = plt.figure(layout='constrained', figsize=(13, 7))
    gs = GridSpec(2, 8, wspace=0.05, hspace=0.05, figure=fig)

    add_plot('one', gs[0, 0:4], '(a) PC1', {'left': 'y', 'top': 'x'})
    add_plot('two', gs[0, 4:8], '(b) PC2', {'right': 'y', 'top': 'x'})
    add_plot('three', gs[1, 0:4], '(c) PC3', {'left': 'y', 'bottom': 'x'})
    add_plot('one and two', gs[1, 4:8], '(d) PC1+PC2', {'right': 'y', 'bottom': 'x'})

    cbar_ax = fig.add_axes([0.15, -0.04, 0.7, 0.02])
    cb = plt.colorbar(ScalarMappable(cmap=custom_cmap,norm=norm), cax=cbar_ax, label='% of explained variance',
                    orientation='horizontal', ticks=levels, spacing='proportional')
    cb.ax.xaxis.set_label_text('% of explained variance')

    plt.savefig(f"{outdir}pca_explainedvar2.png", bbox_inches='tight')
    plt.close()