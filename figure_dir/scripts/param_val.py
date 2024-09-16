import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import os
from matplotlib.lines import Line2D

dir = '/home/mfroelich/Thesis/LSTM_final_param_validation/losses/'
result_dir = '/home/mfroelich/Thesis/figure_dir/plots/'

model_names = ['88_0','88_1e-06','88_0.0001','64_0','64_1e-06','64_0.0001','48_0','48_1e-06','48_0.0001']
train_losses = [np.load(f'{dir}{model_names[x]}_train_losses.npy') for x in range(len(model_names))]
val_losses = [np.load(f'{dir}{model_names[x]}_val_losses.npy') for x in range(len(model_names))]
min_losses = [min(x) for x in val_losses]

def plot_cross_heatmap(min_losses, hidden_units_list, weight_decay_list, save_address):
        fig, ax = plt.subplot(figsize=(12, 12))
        ax.text(0,1,'(a) Acheived validation loss', ha='left', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'))
        sns.heatmap(np.array(min_losses).reshape(3,3), annot=True, fmt=".4f", xticklabels=weight_decay_list, yticklabels=hidden_units_list, cmap="crest")
        plt.ylabel('# Hidden Units')
        plt.xlabel('L2-reg parameter')
        plt.tight_layout()
        plt.savefig(save_address)
        plt.close()

#plot_cross_heatmap(min_losses,[88,64,48],[0,1e-6,1e-4],result_dir+'val_results')

def plot_losses(training_losses, validation_losses):
    """
    Plots training and validation losses on the same axes for any number of models.
    
    Args:
    - training_losses: List of arrays, each containing training losses for a model.
    - validation_losses: List of arrays, each containing validation losses for a model.
    """
    num_models = len(training_losses)
    assert num_models == len(validation_losses), "Number of training and validation models must be the same."
    
    # Define colors for each model (cycled if more models are provided)
    colors = ['#8CA100','#C2D262','#D8DDA8','#A12217','#D66962','#F9B0AB','#1755A1','#619AE1','#9FBAE0']
    line_styles = ['-', '--']  # Solid for training, dashed for validation

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training and validation losses for each model
    for i in range(num_models):
        x_vals_train = np.arange(1, len(training_losses[i]) + 1)
        x_vals_val = np.arange(1, len(validation_losses[i]) + 1)

        # Plot training losses (solid line)
        ax.plot(x_vals_train, training_losses[i], color=colors[i % len(colors)], linestyle=line_styles[0])
        
        # Plot validation losses (dashed line)
        ax.plot(x_vals_val, validation_losses[i], color=colors[i % len(colors)], linestyle=line_styles[1])

    ax.set_xlabel('Epochs')
    ax.set_ylabel(r'Huber-loss ($\delta = 2$)')
    ax.grid(True)

    ax.text(0.09, 0.95, '(b) Loss vs Epochs', ha='left', va='top', transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'))
    
    # Create first legend for colors (different models)
    lines_hidden_units = [Line2D([0], [0], color='#8CA100', lw=2),
                            Line2D([0], [0], color='#DB2F1E', lw=2),
                            Line2D([0], [0], color='#1755A1', lw=2)]
    legend1 = plt.legend(lines_hidden_units, ['88','64','48'], title='# Hidden Units', loc='upper right')
    plt.gca().add_artist(legend1)

    lines_weight_decay = [Line2D([0], [0], color='black', lw=2, alpha=1),
                      Line2D([0], [0], color='black', lw=2, alpha=0.7),
                      Line2D([0], [0], color='black', lw=2, alpha=0.4)]
    legend2 = plt.legend(lines_weight_decay, ['0', '1e-6', '1e-4'], title='L2-reg parameter', loc='upper right', bbox_to_anchor=(0.85,1))
    plt.gca().add_artist(legend2)

    # Create second legend for line styles (Training vs Validation)
    style_lines = [mlines.Line2D([], [], color='black', linestyle=line_styles[0], label='Training'),
                   mlines.Line2D([], [], color='black', linestyle=line_styles[1], label='Validation')]
    ax.legend(handles=style_lines, loc="lower left", title="")

    plt.tight_layout()
    plt.savefig('/home/mfroelich/Thesis/figure_dir/plots/model_losses',bbox_inches='tight')
    plt.close()

#plot_losses(train_losses, val_losses)


def plot_combined_figure(min_losses, training_losses, validation_losses, hidden_units_list, weight_decay_list, save_address):
    """
    Combines the heatmap of validation losses and the plot of training/validation losses into a single figure.
    
    Args:
    - min_losses: List of minimum validation losses for each model.
    - training_losses: List of arrays, each containing training losses for a model.
    - validation_losses: List of arrays, each containing validation losses for a model.
    - hidden_units_list: List of hidden units used in the models.
    - weight_decay_list: List of weight decay (L2-regularization) parameters.
    - save_address: The file path where the combined figure will be saved.
    """
    # Define colors and line styles for the second plot
    colors = ['#8CA100','#C2D262','#D8DDA8','#A12217','#D66962','#F9B0AB','#1755A1','#619AE1','#9FBAE0']
    line_styles = ['-', '--']  # Solid for training, dashed for validation

    # Create a figure with 2 subplots (1x2 grid)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot (a) Heatmap of validation losses
    sns.heatmap(np.array(min_losses).reshape(3, 3), annot=True, fmt=".4f", 
                xticklabels=weight_decay_list, yticklabels=hidden_units_list, 
                cmap="crest", ax=ax1, cbar_kws={'label': r'Huber-Loss ($\delta = 2$)'})
    ax1.set_title('(a) Best validation loss')
    ax1.set_ylabel('# hidden units')
    ax1.set_xlabel('L2-reg parameter')

    ax1.text(0.025,0.975,'(a) Acheived validation loss', ha='left', va='top', transform=ax1.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'))

    # Plot (b) Loss vs Epochs
    num_models = len(training_losses)
    
    # Plot training and validation losses for each model
    for i in range(num_models):
        x_vals_train = np.arange(1, len(training_losses[i]) + 1)
        x_vals_val = np.arange(1, len(validation_losses[i]) + 1)

        # Plot training losses (solid line)
        ax2.plot(x_vals_train, training_losses[i], color=colors[i % len(colors)], linestyle=line_styles[0])
        
        # Plot validation losses (dashed line)
        ax2.plot(x_vals_val, validation_losses[i], color=colors[i % len(colors)], linestyle=line_styles[1])

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(r'Huber-loss ($\delta = 2$)')
    ax2.grid(True)

    ax2.text(0.025, 0.975, '(b) Loss vs Epochs', ha='left', va='top', transform=ax2.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'))

    # Add legends for the second plot
    lines_hidden_units = [Line2D([0], [0], color='#8CA100', lw=2),
                          Line2D([0], [0], color='#DB2F1E', lw=2),
                          Line2D([0], [0], color='#1755A1', lw=2)]
    legend1 = ax2.legend(lines_hidden_units, ['88', '64', '48'], title='# Hidden Units', loc='upper right')
    ax2.add_artist(legend1)

    lines_weight_decay = [Line2D([0], [0], color='black', lw=2, alpha=1),
                          Line2D([0], [0], color='black', lw=2, alpha=0.7),
                          Line2D([0], [0], color='black', lw=2, alpha=0.4)]
    legend2 = ax2.legend(lines_weight_decay, ['0', '1e-6', '1e-4'], title='L2-reg parameter', loc='upper right', bbox_to_anchor=(0.85, 1))
    ax2.add_artist(legend2)

    style_lines = [mlines.Line2D([], [], color='black', linestyle=line_styles[0], label='Training'),
                   mlines.Line2D([], [], color='black', linestyle=line_styles[1], label='Validation')]
    ax2.legend(handles=style_lines, loc="lower left", title="")

    # Save the combined figure
    plt.tight_layout()
    plt.savefig(save_address, bbox_inches='tight')
    plt.close()

# Call the combined plotting function
plot_combined_figure(min_losses, train_losses, val_losses, [88, 64, 48], [0, 1e-6, 1e-4], result_dir+'combined_figure.png')