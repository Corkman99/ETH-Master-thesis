import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

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
    colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink']
    line_styles = ['-', '--']  # Solid for training, dashed for validation

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot training and validation losses for each model
    for i in range(num_models):
        x_vals_train = np.arange(1, len(training_losses[i]) + 1)
        x_vals_val = np.arange(1, len(validation_losses[i]) + 1)

        # Plot training losses (solid line)
        ax.plot(x_vals_train, training_losses[i], color=colors[i % len(colors)], linestyle=line_styles[0], 
                label=f'Model {i+1} Training')
        
        # Plot validation losses (dashed line)
        ax.plot(x_vals_val, validation_losses[i], color=colors[i % len(colors)], linestyle=line_styles[1], 
                label=f'Model {i+1} Validation')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSELoss')
    ax.grid(True)

    # Create first legend for colors (different models)
    model_patches = [mpatches.Patch(color=colors[i % len(colors)], label=f'Model {i+1}') for i in range(num_models)]
    leg1 = ax.legend(handles=model_patches, bbox_to_anchor=(1.0, 1), loc="upper right", title="Models")
    ax.add_artist(leg1)

    # Create second legend for line styles (Training vs Validation)
    style_lines = [mlines.Line2D([], [], color='black', linestyle=line_styles[0], label='Training'),
                   mlines.Line2D([], [], color='black', linestyle=line_styles[1], label='Validation')]
    ax.legend(handles=style_lines, loc="lower right", title="")

    plt.tight_layout()
    plt.savefig('/home/mfroelich/Thesis/figure_dir/plots/model_losses',bbox_inches='tight')
    plt.close()


training_losses = [np.load('/home/mfroelich/Thesis/LSTM_results/all_0.2dropout_1e5lr/losses/64_0_train_losses.npy'), np.load('/home/mfroelich/Thesis/LSTM_results/all_0.3dropout_1e6lr/losses/48_0_train_losses.npy'),
                   np.load('/home/mfroelich/Thesis/LSTM_results/all_0.3dropout_2e6lr/losses/48_1e-06_train_losses.npy'), np.load('/home/mfroelich/Thesis/LSTM_results/all_0.3dropout_2e6lr/losses/64_1e-06_train_losses.npy')]

validation_losses = [np.load('/home/mfroelich/Thesis/LSTM_results/all_0.2dropout_1e5lr/losses/64_0_val_losses.npy'), np.load('/home/mfroelich/Thesis/LSTM_results/all_0.3dropout_1e6lr/losses/48_0_val_losses.npy'),
                     np.load('/home/mfroelich/Thesis/LSTM_results/all_0.3dropout_2e6lr/losses/48_1e-06_val_losses.npy'),np.load('/home/mfroelich/Thesis/LSTM_results/all_0.3dropout_2e6lr/losses/64_1e-06_val_losses.npy')]

# Plot
plot_losses(training_losses, validation_losses)