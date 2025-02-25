import math
import matplotlib.pyplot as plt
import torch
from typing import List, Literal


def format_title(text: str):
    text = text.replace('sn', 'Signal + Noise')
    return text.replace('_', ' ').title()


accept_fields = Literal[
    'input_data', 'target_data', 'predict_data',
    'original', 'original_time', 'original_time_norm',
    'signal', 'signal_time', 'signal_time_norm',
    'noise', 'noise_time', 'noise_time_norm',
    'sn', 'sn_time', 'sn_time_norm',
]

def plot2d(data_list, titles: List[str], cmap='viridis'):
    num = len(data_list)
    ncols = 3
    if num < ncols:
        ncols = num
    nrows = math.ceil(num / ncols)
    figsize = (4.67 * ncols, 4 * nrows)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    im = None
    zmin, zmax = None, None

    for data in data_list:
        zmin = min(zmin, data.min()) if zmin is not None else data.min()
        zmax = max(zmax, data.max()) if zmax is not None else data.max()

    for i, data in enumerate(data_list):
        if type(data) == torch.Tensor:
            data = data.detach().cpu().numpy()
        data = data.squeeze()
        im = ax[i // 3, i % 3].imshow(data, origin='lower', cmap=cmap, vmin=zmin, vmax=zmax)
        ax[i // 3, i % 3].set_xlabel('X')
        ax[i // 3, i % 3].set_ylabel('Y')
        ax[i // 3, i % 3].set_title(format_title(titles[i]))

    for i in range(num, nrows * ncols):
        row, col = divmod(i, ncols)
        fig.delaxes(ax[row, col])

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.05, location='right')
    cbar.set_label('Time')
    return fig, ax


def plot2d_f(data, fields: List[accept_fields], index: int = 0, cmap='viridis'):
    return plot2d([getattr(data, field)[index] for field in fields], fields, cmap)

def plot3d(data_list, titles: List[str], cmap='viridis'):
    num = len(data_list)
    ncols = 3
    if num < ncols:
        ncols = num
    nrows = math.ceil(num / ncols)
    figsize = (5 * ncols, 6 * nrows)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={'projection': '3d'}, squeeze=False)
    im = None
    zmin, zmax = None, None

    for data in data_list:
        zmin = min(zmin, data.min()) if zmin is not None else data.min()
        zmax = max(zmax, data.max()) if zmax is not None else data.max()

    for i, data in enumerate(data_list):
        if type(data) == torch.Tensor:
            data = data.detach().cpu().numpy()
        data = data.squeeze()
        y, x = data.nonzero()
        time_values = data[y, x]
        im = ax[i // 3, i % 3].scatter(x, y, time_values, c=time_values, cmap=cmap, vmin=zmin, vmax=zmax)
        ax[i // 3, i % 3].set_xlabel('X')
        ax[i // 3, i % 3].set_ylabel('Y')
        ax[i // 3, i % 3].set_zlabel('Time', rotation=90)
        ax[i // 3, i % 3].set_title(format_title(titles[i]))
        ax[i // 3, i % 3].view_init(elev=30, azim=-45, roll=0)
        ax[i // 3, i % 3].set_box_aspect((1, 1, 1), zoom=0.8)

    for i in range(num, nrows * ncols):
        row, col = divmod(i, ncols)
        fig.delaxes(ax[row, col])

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.05, location='right')
    cbar.set_label('Time')
    return fig, ax

def plot3d_f(data, fields: List[accept_fields], index: int = 0, cmap='viridis'):
    return plot3d([getattr(data, field)[index] for field in fields], fields, cmap)
