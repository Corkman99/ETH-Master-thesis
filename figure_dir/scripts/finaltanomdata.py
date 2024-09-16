import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import xarray as xr
from matplotlib.ticker import FuncFormatter

num_years = 41
# vicinity of Madrid, Spain (40.5N, 3.5W)
# vicinity of New Dehli, India (28.5N, 77E)
# vicinity of Brazilia, Brazil (16S, 48W)
# Perth, Australia (32S, 116E)
lat = [28.5,45,-32]
lon = [77,-30,116]
label = ['28.5N 77E','45N 30W','32S 116E']
colour = ['#D4AD59','#AB88B8','#59D49F']
address = '/net/litho/atmosdyn/roethlim/data/lastvar/era5/upload/TX1day/data/figures/TX1day_decomposition_era5_v10_final.nc'
outdir = '/home/mfroelich/Thesis/figure_dir/plots/'

ds = xr.open_dataset(address,chunks={'years':1,'lat':181,'lon':361})[['T_anom','adv','adiab','diab']]
ds = ds.mean('lev',skipna=True)

datasets = []
for (la,lo,lab,col) in zip(lat,lon,label,colour):
    ds_i = ds.sel({'lat':la,'lon':lo,'years':range(1980,2021)},drop=True)
    stacked = np.stack([ds_i['T_anom'].values,
                              ds_i['adv'].values,
                              ds_i['adiab'].values,
                              ds_i['diab'].values], axis=-1)
    datasets.append([stacked,lab,col])

titles = ['(a) Adv T\' vs T\'', 
          '(b) Adiab T\' vs T\'', 
          '(c) Diab T\' vs T\'', 
          '(d) Adv T\' vs Adiab T\'', 
          '(e) Adiab T\' vs Diab T\'', 
          '(f) Diab T\' vs Adv T\'']

pairs = [(1, 0), (2, 0), (3, 0), (1, 2), (2, 3), (3, 1)]

# Axis labels
axis_labels = ['T\'', 'Adv T\'', 'Adiab T\'', 'Diab T\'']

# Create the figure and grid spec for better control of layout
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.4], height_ratios=[1, 1, 0.4], hspace=0.05,wspace=0.3)

# Create scatter plots with a histogram on the right only for position [0,2]
for i, (title, (x_idx, y_idx)) in enumerate(zip(titles[:3], pairs[:3])):
    ax = fig.add_subplot(gs[0, i])
    ax.text(0.03, 0.97, title, ha='left', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'))

    # Plot each dataset
    for data, label, color in datasets:
        x = data[:, x_idx]
        y = data[:, y_idx]
        
        # Scatter plot
        ax.scatter(x, y, color=color, alpha=0.7)
        
        max_idx = np.argmax(data[:,0].squeeze())
        ax.scatter(x[max_idx],y[max_idx],color=color,alpha=1,marker='D',s=100)

        # Linear regression
        slope, intercept, _, _, _ = linregress(x, y)
        ax.plot(x, intercept + slope * x, color=color, linestyle='-', linewidth=2)

    # Set labels
    ax.xaxis.set_ticks_position('top')  # Move x-ticks to the top
    ax.xaxis.set_label_position('top')  # Move x-axis label to the top
    ax.tick_params(axis='x', which='both', bottom=False, top=True)  # Show ticks at the top, hide at the bottom
    ax.set_xlabel(axis_labels[x_idx])
    ax.set_ylabel(axis_labels[y_idx])
    ax.yaxis.set_ticks_position('both')

    if i > 0:
        ax.set_yticklabels([])
        ax.set_ylabel('')

    # Add histogram to the right only for the third plot in the top row
    if i == 2:
        hist_ax = fig.add_subplot(gs[0, 3])
        for data, label, color in datasets:
            hist_ax.hist(data[:, y_idx], bins=8, orientation='horizontal', color=color, alpha=0.6)
        hist_ax.set_xlabel('Frequency')
        hist_ax.set_yticklabels([])  # Hide y-tick labels for histogram

        pos = hist_ax.get_position()
        new_pos = [pos.x0-0.038, pos.y0, pos.width, pos.height]
        hist_ax.set_position(new_pos)

# Create scatter plots with inverted histograms below for the second row
for i, (title, (x_idx, y_idx)) in enumerate(zip(titles[3:], pairs[3:])):
    ax = fig.add_subplot(gs[1, i])
    ax.text(0.03, 0.97, title, ha='left', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'))
    
    # Plot each dataset
    for data, label, color in datasets:
        x = data[:, x_idx]
        y = data[:, y_idx]
        
        # Scatter plot
        ax.scatter(x, y, color=color, alpha=0.7)
        
        max_idx = np.argmax(data[:,0].squeeze())
        ax.scatter(x[max_idx],y[max_idx],color=color,alpha=1,marker='D',s=100)

        # Linear regression
        slope, intercept, _, _, _ = linregress(x, y)
        ax.plot(x, intercept + slope * x, color=color, linestyle='-', linewidth=2)

    # Set y-labels normally
    ax.set_ylabel(axis_labels[y_idx])

    # Ensure x-ticks are shown at both top and bottom
    ax.tick_params(axis='x', which='both', bottom=True, top=True)
    ax.xaxis.set_ticks_position('both')
    
    # Remove x-tick labels
    ax.set_xticklabels([])

    # Add inverted histogram below
    hist_ax = fig.add_subplot(gs[2, i]) #, sharex=ax)
    max_count = 0
    for data, label, color in datasets:
        counts, bins, patches = hist_ax.hist(data[:, x_idx], bins=8, color=color, alpha=0.6)
        for patch in patches:
            patch.set_height(-patch.get_height())  # Invert histogram bars
        if max(counts) > max_count:
            max_count = max(counts)

    def magnitude_formatter(x, pos):
        return f'{abs(int(x))}'
    
    hist_ax.set_ylim(bottom=-max_count*1.1, top=0)
    hist_ax.yaxis.set_major_formatter(FuncFormatter(magnitude_formatter))
    hist_ax.set_ylabel('Frequency')

    hist_ax.set_xlabel(axis_labels[x_idx])
    
    # Ensure x-ticks are shown at both top and bottom for histograms
    hist_ax.tick_params(axis='x', which='both', bottom=True, top=True)
    hist_ax.xaxis.set_ticks_position('both')
    hist_ax.xaxis.set_tick_params(labelbottom=True, labeltop=False)

    #pos = hist_ax.get_position()
    #new_pos = [pos.x0, pos.y0+0.01, pos.width, pos.height]
    #hist_ax.set_position(new_pos)

# --- Legend Placement ---
# Create custom legend handles
handles = [plt.Line2D([], [], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
           for _, label, color in datasets]

# Add legend to the right of the plot at position [1,2]
legend_ax = fig.add_subplot(gs[1, 3])
legend_ax.axis('off')  # Hide the axis
pos = legend_ax.get_position()
new_pos = [pos.x0-0.038, pos.y0-0.02, pos.width, pos.height]
legend_ax.set_position(new_pos)

# Add legend to this axis
legend_ax.legend(handles=handles, loc='center', frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{outdir}flat_data.png',bbox_inches='tight')
plt.close()
