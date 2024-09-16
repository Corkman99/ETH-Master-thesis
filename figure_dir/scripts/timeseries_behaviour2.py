import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from scipy.signal import correlate
import matplotlib.gridspec as gridspec

from matplotlib import colors as mcolors

import xarray as xr

num_years = 41
lags = 32
# vicinity of Madrid, Spain (40.5N, 3.5W)
# vicinity of New Dehli, India (28.5N, 77E)
# vicinity of Brazilia, Brazil (16S, 48W)

# mid-atlantic ocean (45N, 30W)
# vicinity of Santa Cruz de la Sierra, Bolivia (17.5S, 63.5W)
lat = -16
lon = -48
address = '/net/litho/atmosdyn2/mfroelich/TS_TX1day_mean-lvl'
outdir = '/home/mfroelich/Thesis/figure_dir/plots/'

ds = xr.open_dataset(address,chunks={'year':1,'lat':181,'lon':361,'trajtime':121})[['T_anom','adv','adiab','diab']]
ds = ds.fillna(0)
ds = ds.sel({'lat':lat,'lon':lon},drop=True)
ds = ds.sel({'year':list(np.arange(1980,2021,1))})
ds = ds.diff(dim='trajtime')

data = np.stack([ds['T_anom'].values,
                 ds['adv'].values,
                 ds['adiab'].values,
                 ds['diab'].values], axis=-1)

# Compute global min and max for y-axis limits
global_min = data.min()
global_max = data.max()

# Calculate mean and quantiles across the first dimension (40 timeseries)
mean = np.mean(data, axis=0)              # Shape: (120, 4)
quantile_25 = np.quantile(data, 0.25, axis=0)  # Shape: (120, 4)
quantile_75 = np.quantile(data, 0.75, axis=0)  # Shape: (120, 4)

# Identify the index of the timeseries with the largest final value in the first graph (dimension 0)
max_idx = np.argmax(data[:, -1, 0])

# Create a figure with a specified size
fig = plt.figure(figsize=(16, 18))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])

# Define axes for timeseries plots
ax_ts_1 = fig.add_subplot(gs[0, 0])
ax_ts_2 = fig.add_subplot(gs[0, 1])
ax_ts_3 = fig.add_subplot(gs[1, 0])
ax_ts_4 = fig.add_subplot(gs[1, 1])

# Define axes for correlation plots
ax_auto = fig.add_subplot(gs[2, 0])
ax_cross = fig.add_subplot(gs[2, 1])

# List of axes for easier iteration
timeseries_axes = [ax_ts_1, ax_ts_2, ax_ts_3, ax_ts_4]
names = ['(a) Event T\'','(b) Adv T\'','(c) Adiab T\'','(d) Diab T\'']

x_domain = np.linspace(10,1,num=120)
x_ticks = [x_domain[y] for y in [0, 39, 79, 95, 107, 111, 115, 119]]
x_labels = ['15 d', '10 d', '5 d', '3 d', '36 h', '24 h', '12 h', '0 h']

# Plot the four timeseries graphs
for i in range(4):
    ax = timeseries_axes[i]
    
    # Plot all 40 timeseries in grey
    for j in range(data.shape[0]):
        ax.plot(x_domain,data[j, :, i], color='grey', alpha=0.5)
    
    # Highlight the selected timeseries in red
    ax.plot(x_domain,data[max_idx, :, i], color='red', linewidth=2,label='Max event T\'(0)') 
    
    ax.fill_between(x_domain,quantile_25[:, i], quantile_75[:, i], color='lightblue', alpha=0.5, label='25th-75th Quantile Range')
    ax.plot(x_domain,mean[:, i], color='black', linewidth=2, label='Mean')
    
    # Set titles and labels
    ax.text(0.025, 0.95, names[i], ha='left', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'))
    ax.set_xlabel('Time to event (h)')
    ax.set_ylabel('T\'')
    
    # Set y-axis limits
    ax.set_ylim(global_min, global_max)
    ax.set_xscale('log')
    ax.invert_xaxis()

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    # Optional: Add legend only to the first plot to avoid clutter
    if i == 0:
        ax.legend(loc='lower right')

def auto_correlation(data, max_lag):
    num_samples, num_trajectories, num_variables = data.shape
    lag_range = 2 * max_lag + 1
    avg_auto_corr = np.full((lag_range, num_variables), np.nan)

    for sample in range(num_samples):
        # Find the index of the first non-zero element in the trajectory for each sample
        first_non_zero_index = np.argmax(data[sample, :, 0] != 0)
        
        # If all elements are zero, handle it by setting a large index
        if data[sample, first_non_zero_index:, 0].size == 0:
            first_non_zero_index = num_trajectories
        
        for var in range(num_variables):
            series = data[sample, first_non_zero_index:, var]
            if series.size == 0:
                continue
            
            # Compute the auto-correlation
            corr = np.correlate(series, series, mode='full')
            middle = len(corr) // 2
            corr = corr[middle: middle + lag_range] / (np.var(series) * len(series))
            
            # Handle lags larger than series length by padding with NaNs
            if len(corr) < lag_range:
                corr = np.concatenate((corr, np.full(lag_range - len(corr), np.nan)))
            
            # Assign the result
            avg_auto_corr[:, var] = np.nanmean([avg_auto_corr[:, var], corr], axis=0)

    return avg_auto_corr

def auto_correlation_with_final(data,max_lag):
    num_samples, num_trajectories, num_variables = data.shape
    avg_auto_corr = np.full((max_lag+1, num_variables), np.nan)

    first_non_zero_index = np.argmax(data[:, :, 0] != 0, axis=1)

    for i in range(max_lag+1):
        subset = first_non_zero_index <= num_trajectories - i
        if subset.sum() <= 5: # if there are less than 10 samples, leave as NA
            continue
        xt = data[subset,-(1+i),:].squeeze()
        xT = data[subset,-1,:].squeeze()

        for var in range(num_variables):
            avg_auto_corr[i, var] = np.corrcoef(xT[:,var], xt[:,var])[0,1]

    return avg_auto_corr

def cross_correlation_with_final(data,max_lag):
    num_samples, num_trajectories, num_variables = data.shape
    avg_cross_corr = np.full((max_lag+1, num_variables-1), np.nan)

    first_non_zero_index = np.argmax(data[:, :, 0] != 0, axis=1)

    for i in range(max_lag+1):
        subset = first_non_zero_index <= num_trajectories - i
        if subset.sum() <= 5: # if there are less than 10 samples, leave as NA
            continue
        xt = data[subset,-(1+i),-3:].squeeze()
        xT = data[subset,-1,0].squeeze()

        for var in range(num_variables-1):
            avg_cross_corr[i, var] = np.corrcoef(xT, xt[:,var])[0,1]

    return avg_cross_corr

def cross_correlation(data, max_lag):
    num_samples, num_trajectories, num_variables = data.shape
    lag_range = 2 * max_lag + 1
    avg_cross_corr = np.full((lag_range, num_variables - 1), np.nan)

    for sample in range(num_samples):
        # Find the index of the first non-zero element in the trajectory for each sample
        first_non_zero_index = np.argmax(data[sample, :, 0] != 0)
        
        # If all elements are zero, handle it by setting a large index
        if data[sample, first_non_zero_index:, 0].size == 0:
            first_non_zero_index = num_trajectories
        
        for var in range(1, num_variables):
            series1 = data[sample, first_non_zero_index:, 0]
            series2 = data[sample, first_non_zero_index:, var]
            
            if series1.size == 0 or series2.size == 0:
                continue
            
            # Compute the cross-correlation
            corr = np.correlate(series1, series2, mode='full')
            middle = len(corr) // 2
            corr = corr[middle - max_lag: middle + max_lag + 1] / (np.sqrt(np.var(series1) * np.var(series2)) * len(series1))
            
            # Handle lags larger than series length by padding with NaNs
            if len(corr) < lag_range:
                padding = np.full(lag_range - len(corr), np.nan)
                corr = np.concatenate((padding[:max_lag], corr, padding[max_lag:]))
            
            # Assign the result
            avg_cross_corr[:, var - 1] = np.nanmean([avg_cross_corr[:, var - 1], corr], axis=0)

    return avg_cross_corr

auto_corrs = auto_correlation_with_final(data, max_lag=lags)
cross_corrs = cross_correlation_with_final(data, max_lag=lags)

x_auto = np.arange(0,lags+1,1)
x_cross = np.arange(0,lags+1,1)

colors = ['#D36159','#4B9C50','#D1BA48','#5994D3']

# Plot Auto-Correlation
bar_width = 0.2

labels = ['Event T\'', 'Adv T\'', 'Adiab T\'', 'Diab T\'']

for i in range(auto_corrs.shape[1]):
    dat = list(auto_corrs[:,i])
    dat.reverse()
    ax_auto.bar(x_auto + i * 0.2, dat, width=bar_width, label=labels[i],color = colors[i])

ax_auto.text(0.025, 0.1, r'(e) Auto-correlation', ha='left', va='top', transform=ax_auto.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'))
ax_auto.set_xlabel(r'Lag $k$ (days)')
ax_auto.set_ylabel(r"$\text{Corr}(X^{(T)},X^{(T-k)})$")
ax_auto.set_xlim((-1,lags+1))
ax_auto.set_ylim((auto_corrs.min()-0.1,1))
ax_auto.set_xticks([0,8,16,24,32])
ax_auto.set_xticklabels(['4','3','2','1','0'])
ax_auto.legend(loc='upper left')

for i in range(cross_corrs.shape[1]):
    dat = list(cross_corrs[:,i])
    dat.reverse()
    ax_cross.bar(x_cross + i * 0.2, dat, width=bar_width, color=colors[i+1])

ax_cross.text(0.025, 0.1, '(f) Cross-correlation', ha='left', va='top', transform=ax_cross.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'))
ax_cross.set_xlabel(r'Lag $k$ (days)')
ax_cross.set_ylabel(r"$\text{Corr}(\text{event} T'^{(T)},X^{(T-k)})$")
ax_cross.set_xlim((-1,lags+1))
ax_cross.set_ylim((cross_corrs.min()-0.1,cross_corrs.max()+0.1))
ax_cross.set_xticks([0,8,16,24,32])
ax_cross.set_xticklabels(['4','3','2','1','0'])

# Adjust layout for better spacing
plt.tight_layout()

# Adjust layout
plt.savefig(f'{outdir}timeseries2_diff_lat{lat}_lon{lon}.png') #, bbox_inches='tight')
plt.close()
