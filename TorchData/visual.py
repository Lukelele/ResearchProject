import numpy as np
import matplotlib.pyplot as plt

# --------- Data Visualisation -----------
FIG_SIZE_2D = (14, 4)
CBAR_ORIENTATION_2D = 'vertical'
CBAR_LOCATION_2D = 'right'

FIG_SIZE_3D = (15, 6)
CBAR_ORIENTATION_3D = 'vertical'
CBAR_LOCATION_3D = 'right'

def plot2d(data, ax: plt.Axes = None, z_lim=(None, None)):
    if ax is None:
        ax = plt.figure().add_subplot()
    im = ax.imshow(data, cmap='viridis', vmin=z_lim[0], vmax=z_lim[1])
    ax.set_xlabel('Y')
    ax.set_ylabel('X')

    return im

def plot3d(data, ax: plt.Axes = None, elev=30, azim=-45, roll=0, aspect=(1, 1, 1), zoom=0.8, z_lim=(None, None)):
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
    data_np = data.numpy().squeeze()
    y, x = np.nonzero(data_np)
    time_values = data_np[y, x]

    im = ax.scatter(y, x, time_values, c=time_values, cmap='viridis', vmin=z_lim[0], vmax=z_lim[1])
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Time', rotation=90)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_box_aspect(aspect, zoom=zoom)
    if z_lim != (None, None):
        ax.set_zlim(z_lim)

    return im

def compare_plot2d(clean_data, noisy_data, pred_data,
                   figsize=FIG_SIZE_2D, cbar_orientation=CBAR_ORIENTATION_2D, cbar_location=CBAR_LOCATION_2D):
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    z_max = max(clean_data.max(), noisy_data.max(), pred_data.max()) + 10
    z_min = max(min(clean_data.min(), noisy_data.min(), pred_data.min()) - 10, 0)
    im = plot2d(clean_data, ax[0], z_lim=(z_min, z_max))
    ax[0].set_title("Original Image")
    plot2d(noisy_data, ax[1], z_lim=(z_min, z_max))
    ax[1].set_title("Noisy Image")
    plot2d(pred_data, ax[2], z_lim=(z_min, z_max))
    ax[2].set_title("Reconstructed Image")
    cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation, fraction=0.03, pad=0.05, location=cbar_location)
    cbar.set_label('Time')

def compare_plot3d(clean_data, noisy_data, pred_data,
                   figsize=FIG_SIZE_3D, cbar_orientation=CBAR_ORIENTATION_3D, cbar_location=CBAR_LOCATION_3D,
                   elev=30, azim=-45, roll=0, aspect=(1, 1, 1), zoom=0.8):
    z_max = max(clean_data.max(), noisy_data.max(), pred_data.max()) + 10
    z_min = max(min(clean_data.min(), noisy_data.min(), pred_data.min()) - 10, 0)
    z_lim = (z_min, z_max)
    fig, ax = plt.subplots(1, 3, figsize=figsize,
                             subplot_kw={'projection': '3d'})
    im = plot3d(clean_data, ax[0], elev=elev, azim=azim, roll=roll, aspect=aspect, zoom=zoom, z_lim=z_lim)
    ax[0].set_title("Original Image")
    plot3d(noisy_data, ax[1], elev=elev, azim=azim, roll=roll, aspect=aspect, zoom=zoom, z_lim=z_lim)
    ax[1].set_title("Noisy Image")
    plot3d(pred_data, ax[2], elev=elev, azim=azim, roll=roll, aspect=aspect, zoom=zoom, z_lim=z_lim)
    ax[2].set_title("Reconstructed Image")

    cbar = fig.colorbar(im, ax=ax, orientation=cbar_orientation, fraction=0.01, pad=0.05, location=cbar_location)
    cbar.set_label('Time')

def compare_plot(clean_data, noisy_data, pred_data,
                 figsize_2d=FIG_SIZE_2D, cbar_orientation_2d=CBAR_ORIENTATION_2D, cbar_location_2d=CBAR_LOCATION_2D,
                 figsize_3d=FIG_SIZE_3D, cbar_orientation_3d=CBAR_ORIENTATION_3D, cbar_location_3d=CBAR_LOCATION_3D,
                 elev=30, azim=-45, roll=0, aspect=(1, 1, 1), zoom=0.8):
    compare_plot2d(clean_data, noisy_data, pred_data, figsize=figsize_2d, cbar_orientation=cbar_orientation_2d, cbar_location=cbar_location_2d)
    compare_plot3d(clean_data, noisy_data, pred_data, elev=elev, azim=azim, roll=roll, aspect=aspect, zoom=zoom,
                   figsize=figsize_3d, cbar_orientation=cbar_orientation_3d, cbar_location=cbar_location_3d)

def fast_compare_plot2d(test_dataset, pred, index,
                        figsize=FIG_SIZE_2D, cbar_orientation=CBAR_ORIENTATION_2D, cbar_location=CBAR_LOCATION_2D):
    compare_plot2d(test_dataset.signal_time[index].squeeze(0), test_dataset.sn_time[index].squeeze(0), pred[index].squeeze(0),
                   figsize=figsize, cbar_orientation=cbar_orientation, cbar_location=cbar_location)

def fast_compare_plot3d(test_dataset, pred, index,
                        figsize=FIG_SIZE_3D, cbar_orientation=CBAR_ORIENTATION_3D, cbar_location=CBAR_LOCATION_3D,
                        elev=30, azim=-45, roll=0, aspect=(1, 1, 1), zoom=0.8):
    compare_plot3d(test_dataset.signal_time[index].squeeze(0), test_dataset.sn_time[index].squeeze(0), pred[index].squeeze(0),
                   figsize=figsize, cbar_orientation=cbar_orientation, cbar_location=cbar_location,
                   elev=elev, azim=azim, roll=roll, aspect=aspect, zoom=zoom)

def fast_compare_plot(test_dataset, pred, index,
                      figsize_2d=FIG_SIZE_2D, cbar_orientation_2d=CBAR_ORIENTATION_2D, cbar_location_2d=CBAR_LOCATION_2D,
                      figsize_3d=FIG_SIZE_3D, cbar_orientation_3d=CBAR_ORIENTATION_3D, cbar_location_3d=CBAR_LOCATION_3D,
                      elev=30, azim=-45, roll=0, aspect=(1, 1, 1), zoom=0.8):
    compare_plot(test_dataset.signal_time[index].squeeze(0), test_dataset.sn_time[index].squeeze(0), pred[index].squeeze(0),
                 figsize_2d=figsize_2d, cbar_orientation_2d=cbar_orientation_2d, cbar_location_2d=cbar_location_2d,
                 figsize_3d=figsize_3d, cbar_orientation_3d=cbar_orientation_3d, cbar_location_3d=cbar_location_3d,
                 elev=elev, azim=azim, roll=roll, aspect=aspect, zoom=zoom)
