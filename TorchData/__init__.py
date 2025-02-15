import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt



# --------- Data Generation -----------

def draw_parabola(canvas, origin=None):
    if origin is None:
        origin_x = np.random.randint(0, canvas.shape[1])
        origin_y = np.random.randint(canvas.shape[0]*0.25, canvas.shape[0])
    else:
        origin_x = origin[0]
        origin_y = origin[1]
    
    theta = np.arange(0, 1*np.pi, 0.0001)
    
    min_side = np.minimum(canvas.shape[1], canvas.shape[0])
    radius = np.random.randint(min_side, min_side*3)
    x = origin_x + radius * np.cos(theta)
    y = (origin_y - radius + 1) + radius * np.sin(theta)
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    
    valid_indices = np.where((x >= 0) & (x < canvas.shape[1]) & (y >= 0) & (y < canvas.shape[0]))
    
    x_parabola_coords = x[valid_indices]
    y_parabola_coords = y[valid_indices]
    canvas[y_parabola_coords, x_parabola_coords] = 1
    
    point_r = (x_parabola_coords[0], y_parabola_coords[0])
    point_l = (x_parabola_coords[-1], y_parabola_coords[-1])
    
    return point_r, point_l


def draw_line(canvas, start, angle, max_lit_px=400):
    angle = np.radians(angle)
    
    dx = np.cos(angle)
    dy = np.sin(angle)

    x, y = start
    lit_pixels = 0
    
    while lit_pixels < max_lit_px:
        x_rounded = int(round(x))
        y_rounded = int(round(y))
        
        if 0 <= x_rounded < canvas.shape[1] and 0 <= y_rounded < canvas.shape[0]:
            canvas[y_rounded, x_rounded] = 1
        
        if lit_pixels != 0:
            if x_rounded <= 0:
                dx = abs(dx)
            elif x_rounded >= canvas.shape[1] - 1:
                dx = -abs(dx)
        
        x += dx
        y -= dy
        lit_pixels += 1


def random_remove_points(canvas, n_points):
    none_zero_indices = canvas.nonzero()
    if none_zero_indices.shape[0] < n_points:
        return

    if n_points >= 1:
        random_indices = np.random.choice(none_zero_indices.shape[0], n_points, replace=False)
    else:
        n = int(n_points * none_zero_indices.shape[0])
        random_indices = np.random.choice(none_zero_indices.shape[0], n, replace=False)

    for i in random_indices:
        x = none_zero_indices[i][1]
        y = none_zero_indices[i][0]
        canvas[y, x] = 0

        
def add_time_value(canvas, t_start):
    rows, cols = canvas.shape

    first_row = 0
    
    for row in range(rows-1, -1, -1):
        non_zero_indices = np.where(canvas[row, :] != 0)[0]
        if non_zero_indices.size > 0:
            if first_row is None:
                first_row = row
            time_index = first_row - row + t_start

            canvas[row, non_zero_indices] = time_index
    
    return canvas


def add_time_dim(canvas, time_index):
    timed_data = torch.zeros(2, canvas.shape[0], canvas.shape[1])
    nonzero_x = canvas.nonzero()[:, 1]
    nonzero_y = canvas.nonzero()[:, 0]
    time_values = canvas[nonzero_y, nonzero_x].to(torch.int)
    timed_data[time_values, nonzero_y, nonzero_x] = 1
    return timed_data

def generate_binary_noise(*dim, p=0.001, magnitude=1):
    random_tensor = torch.rand(*dim)
    return (random_tensor < p).float() * magnitude

def generate_noise(data, p=0.001):
    return torch.clamp(generate_binary_noise(data.shape, p=p, magnitude=1) - data, 0, 1)


def set_continuous_time(canvas, t_start):
    rows, cols = canvas.shape
    canvas = canvas.clone()

    first_row = 0
    time_end = 0

    for row in range(rows-1, -1, -1):
        non_zero_indices = np.where(canvas[row, :] != 0)[0]
        if non_zero_indices.size > 0:
            if first_row is None:
                first_row = row
            time_index = rows - row + t_start
            canvas[row, non_zero_indices] = time_index
            time_end = time_index

    return canvas, time_end


def set_random_time(canvas, t_start, t_end):
    canvas = canvas.clone()

    random_times = torch.randint(t_start, t_end, canvas.shape, dtype=canvas.dtype, device=canvas.device)

    canvas[canvas != 0] = random_times[canvas != 0]

    return canvas



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


# --------- Torch Data Class -----------
class TORCHData():
    def __init__(self, t_dim, x_dim, y_dim, n_remove=10):
        """
        Initiate torch datasets.
        Args:
            t_dim: int, time dimension
            x_dim: int, x dimension
            y_dim: int, y dimension
            n_remove: int, number of points to remove
        """
        self.t_dim = t_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_remove = n_remove
        self.signal = torch.zeros(self.x_dim, self.y_dim)
        self.noise = torch.zeros(self.x_dim, self.y_dim)
        self.sn = torch.zeros(self.x_dim, self.y_dim)
        self.signal_time = torch.zeros(self.x_dim, self.y_dim)
        self.noise_time = torch.zeros(self.x_dim, self.y_dim)
        self.sn_time = torch.zeros(self.x_dim, self.y_dim)
        self.generate()


    def generate(self):
        """
        Generate datasets.
        """
        pr, pl = draw_parabola(self.signal)

        angle = np.random.randint(10, 40)
        draw_line(self.signal, pr, angle)
        draw_line(self.signal, pl, angle)
        random_remove_points(self.signal, self.n_remove)
        self.signal_time, time_end = set_continuous_time(self.signal, 100)
        self.noise = generate_noise(self.signal, p=0.1)
        self.noise_time = set_random_time(self.noise, 100, time_end+10)
        self.sn = self.signal + self.noise
        self.sn_time = self.signal_time + self.noise_time



class TORCHDataset(Dataset):
    def __init__(self, t=100, x=120, y=92, n_remove=50, num_data = 1):
        data = np.array([TORCHData(t, x, y, n_remove) for _ in range(num_data)])

        self.sn_time = [] # signal + noise, (time value)
        self.signal_time = [] # signal (time value)

        for i in data:
            self.sn_time.append(i.sn_time.unsqueeze(0))
            self.signal_time.append(i.signal_time.unsqueeze(0))

        self.sn_time = torch.stack(self.sn_time)
        self.signal_time = torch.stack(self.signal_time)

    def dataloader(self, batch_size=1, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.sn_time)

    def __getitem__(self, idx):
        return torch.tensor(self.sn_time[idx]), torch.tensor(self.signal_time[idx])


class TORCHDataset2Channel(Dataset):
    def __init__(self, t=100, x=120, y=92, n_remove=50, num_data = 1):
        data = np.array([TORCHData(t, x, y, n_remove) for _ in range(num_data)])

        self.sn_time = []
        self.signal_time = []

        for i in data:
            self.sn_time.append(torch.cat((i.sn.unsqueeze(0), i.sn_time.unsqueeze(0)), dim=0))
            self.signal_time.append(torch.cat((i.signal.unsqueeze(0), i.signal_time.unsqueeze(0)), dim=0))

        self.sn_time = torch.stack(self.sn_time)
        self.signal_time = torch.stack(self.y)

    def dataloader(self, batch_size=1, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.sn_time)

    def __getitem__(self, idx):
        return torch.tensor(self.sn_time[idx]), torch.tensor(self.signal_time[idx])
