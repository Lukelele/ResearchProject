import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from numba import njit, prange
from .visual import *
from .metric import *


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

@njit
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

def remove_signal(canvas, n_points):
    non_zero_indices = np.argwhere(canvas != 0)
    if len(non_zero_indices) == 0:
        return
    
    if n_points == -1:
        return

    if n_points >= 1:
        n_remove = min(n_points, len(non_zero_indices))
    else:
        n_remove = int(n_points * len(non_zero_indices))

    remove_indices = np.random.choice(len(non_zero_indices), n_remove, replace=False)
    for idx in remove_indices:
        y, x = non_zero_indices[idx]
        canvas[y, x] = 0

def retain_signal(canvas, n_points):
    non_zero_indices = np.argwhere(canvas != 0)
    if len(non_zero_indices) == 0:
        return
    
    if n_points == -1:
        return

    if n_points >= 1:
        if len(non_zero_indices) < n_points:
            n_remove = 0
            print("Warning: n_points is greater than the number of non-zero points.")
        else:
            n_remove = len(non_zero_indices) - n_points
    else:
        n_remove = int((1 - n_points) * len(non_zero_indices))

    remove_indices = np.random.choice(len(non_zero_indices), n_remove, replace=False)
    for idx in remove_indices:
        y, x = non_zero_indices[idx]
        canvas[y, x] = 0

def add_time_dim(canvas, time_index):
    timed_data = np.zeros(2, canvas.shape[0], canvas.shape[1])
    nonzero_x = canvas.nonzero()[:, 1]
    nonzero_y = canvas.nonzero()[:, 0]
    time_values = canvas[nonzero_y, nonzero_x].to(np.int64)
    timed_data[time_values, nonzero_y, nonzero_x] = 1
    return timed_data

def generate_binary_noise(*dim, p=0.001, magnitude=1):
    random_tensor = np.random.rand(*dim)
    return (random_tensor < p).astype(np.float32) * magnitude

def generate_noise(data, p=0.001):
    return np.clip(generate_binary_noise(*data.shape, p=p, magnitude=1) - data, 0, 1)

@njit
def set_continuous_time(canvas, t_start):
    rows, cols = canvas.shape
    canvas = canvas.copy()

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
    canvas = canvas.copy()

    random_times = np.random.randint(t_start, t_end, canvas.shape)

    canvas[canvas != 0] = random_times[canvas != 0]

    return canvas

# Sellmeier equation coefficients for quartz
B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
C1, C2, C3 = (0.0684043)**2, (0.1162414)**2, (9.896161)**2

def refractive_index(lambda_um):
    """Calculate refractive index using the Sellmeier equation."""
    lambda_sq = lambda_um**2
    return np.sqrt(1 + (B1 * lambda_sq) / (lambda_sq - C1) +
                      (B2 * lambda_sq) / (lambda_sq - C2) +
                      (B3 * lambda_sq) / (lambda_sq - C3))

selected_wavelength = 0.4  # µm
n_selected = refractive_index(selected_wavelength)

# Define dispersion step size based on refractive index
step_size = n_selected * 1  # Scale for visualization


def monte_carlo_dispersion(data, num_steps=50, step_size=1):
    x_index_max, y_index_max = data.shape[0] - 1, data.shape[1] - 1

    photon_coords = np.argwhere(data != 0)
    photon_values = data[data != 0]

    dx = np.random.normal(0, step_size, size=len(photon_coords))
    dy = np.random.normal(0, step_size, size=len(photon_coords))

    new_x = np.clip(photon_coords[:, 0] + dx, 0, x_index_max).astype(int)
    new_y = np.clip(photon_coords[:, 1] + dy, 0, y_index_max).astype(int)

    new_data = np.zeros_like(data)
    new_data[new_x, new_y] = photon_values

    return new_data


# --------- Torch Data Class -----------
class TORCHData:
    def __init__(self, t_dim, x_dim, y_dim, signal_count=1, noise_density=0.1):
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
        self.signal_count = signal_count
        shape = (self.x_dim, self.y_dim)
        self.original = np.zeros(shape, dtype=np.float32)
        self.original_time = np.zeros(shape, dtype=np.float32)
        self.signal = np.zeros(shape, dtype=np.float32)
        self.noise = np.zeros(shape, dtype=np.float32)
        self.noise_density = noise_density
        self.sn = np.zeros(shape, dtype=np.float32)
        self.signal_time = np.zeros(shape, dtype=np.float32)
        self.noise_time = np.zeros(shape, dtype=np.float32)
        self.sn_time = np.zeros(shape, dtype=np.float32)
        self.generate()
        self._to_tensor()

    def generate(self):
        """
        Generate datasets.
        """
        pr, pl = draw_parabola(self.signal)

        angle = np.random.randint(10, 40)
        draw_line(self.signal, pr, angle)
        draw_line(self.signal, pl, angle)
        self.original = self.signal.copy()
        retain_signal(self.signal, self.signal_count)
        self.signal = monte_carlo_dispersion(self.signal, num_steps=50, step_size=step_size)
        self.original_time, time_end = set_continuous_time(self.original, 100)
        self.signal_time, time_end = set_continuous_time(self.signal, 100)
        self.noise = generate_noise(self.signal, p=self.noise_density)
        self.noise_time = set_random_time(self.noise, 100, time_end+10)
        self.sn = self.signal + self.noise
        self.sn_time = self.signal_time + self.noise_time

    def _to_tensor(self):
        self.original = torch.tensor(self.original, dtype=torch.float32)
        self.original_time = torch.tensor(self.original_time, dtype=torch.float32)
        self.signal = torch.tensor(self.signal, dtype=torch.float32)
        self.noise = torch.tensor(self.noise, dtype=torch.float32)
        self.sn = torch.tensor(self.sn, dtype=torch.float32)
        self.signal_time = torch.tensor(self.signal_time, dtype=torch.float32)
        self.noise_time = torch.tensor(self.noise_time, dtype=torch.float32)
        self.sn_time = torch.tensor(self.sn_time, dtype=torch.float32)


class TORCHDataset(Dataset):
    def __init__(self, t=100, x=120, y=92, num_data = 1, signal_count=-1, noise_density=0.1):
        data = np.array([TORCHData(t, x, y, signal_count, noise_density) for _ in range(num_data)])

        self.sn_time = [] # signal + noise, (time value)
        self.signal_time = [] # signal (time value)
        self.signal = []
        self.original = []
        self.original_time = []

        for i in data:
            self.sn_time.append(i.sn_time.unsqueeze(0))
            self.signal_time.append(i.signal_time.unsqueeze(0))
            self.signal.append(i.signal.unsqueeze(0))
            self.original_time.append(i.original_time.unsqueeze(0))
            self.original.append(i.original.unsqueeze(0))

        self.sn_time = torch.stack(self.sn_time)
        self.signal_time = torch.stack(self.signal_time)
        self.signal = torch.stack(self.signal)
        self.original_time = torch.stack(self.original_time)
        self.original = torch.stack(self.original)

    def dataloader(self, batch_size=1, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.sn_time)

    def __getitem__(self, idx):
        return self.sn_time[idx].clone().detach(), self.original_time[idx].clone().detach()


class TORCHDataset2Channel(Dataset):
    def __init__(self, t=100, x=120, y=92, num_data = 1, signal_count=-1, noise_density=0.1):
        data = np.array([TORCHData(t, x, y, signal_count, noise_density) for _ in range(num_data)])

        self.x = []
        self.y = []
        self.original = []
        self.original_time = []
        self.sn_time = []  # signal + noise, (time value)
        self.signal_time = []  # signal (time value)
        self.signal = []

        for i in data:
            self.x.append(torch.cat((i.sn.unsqueeze(0), i.sn_time.unsqueeze(0)), dim=0))
            self.y.append(torch.cat((i.original.unsqueeze(0), i.original_time.unsqueeze(0)), dim=0))
            self.sn_time.append(i.sn_time.unsqueeze(0))
            self.signal_time.append(i.signal_time.unsqueeze(0))
            self.signal.append(i.signal.unsqueeze(0))
            self.original_time.append(i.original_time.unsqueeze(0))
            self.original.append(i.original.unsqueeze(0))

        self.x = torch.stack(self.x)
        self.y = torch.stack(self.y)
        self.sn_time = torch.stack(self.sn_time)
        self.signal_time = torch.stack(self.signal_time)
        self.signal = torch.stack(self.signal)
        self.original_time = torch.stack(self.original_time)
        self.original = torch.stack(self.original)

    def dataloader(self, batch_size=1, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].clone().detach(), self.y[idx].clone().detach()

