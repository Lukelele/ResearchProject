from typing import Literal
import numpy as np
from torch.utils.data import DataLoader, Dataset
from .visual import *
from .metric import *

def draw_on_canvas(canvas, x_coords, y_coords, num_points, value):
    """
    Draws points on a 3D canvas, and returns a mask of valid points.
    Args:
        canvas (np.ndarray): A 3D numpy array (n, y, x) where the points are drawn.
        x_coords (np.ndarray): (n, num_points) array containing x-coordinates to draw.
        y_coords (np.ndarray): (n, num_points) array containing y-coordinates to draw.
        num_points (int): Number of points to draw per layer.
        value (int): Value to draw on the canvas.

    Returns:
        np.ndarray: A boolean mask of valid points.

    """
    n, y, x = canvas.shape
    x_coords = np.round(x_coords).astype(int)
    y_coords = np.round(y_coords).astype(int)

    # Ensure points are within canvas bounds
    valid_mask = (x_coords >= 0) & (x_coords < x) & (y_coords >= 0) & (y_coords < y)

    x_parabola_coords = x_coords[valid_mask]
    y_parabola_coords = y_coords[valid_mask]
    n_parabola_coords = np.arange(n).reshape(-1, 1).repeat(num_points, axis=1)[valid_mask]

    canvas[n_parabola_coords, y_parabola_coords, x_parabola_coords] = value
    return valid_mask


def draw_parabola(canvas, origins=None, resolution_points=None):
    """
    Draws a parabola on each layer of a 3D canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array (n, y, x) where the parabolas are drawn.
        origins (np.ndarray): (n, 2) array containing (x, y) start points per layer.
        resolution_points (int): Number of points to sample along the parabola.

    Returns:
        np.ndarray: (n, 2) array containing (x, y) start and end points per layer.
    """
    n, y, x = canvas.shape
    min_side = min(x, y)

    if origins is None:
        # Randomly sample origins
        origins_x = np.random.randint(0, x, size=(n, 1))
        origins_y = np.random.randint(0, int(y * 0.75), size=(n, 1))
    else:
        origins_x = origins[:, 0]
        origins_y = origins[:, 1]

    if resolution_points is None:
        resolution_points = 10 * min_side

    theta = np.linspace(0, np.pi, resolution_points)  # Precompute theta values

    radius = np.random.uniform(min_side, min_side * 3, size=(n, 1))
    x_coords = origins_x + (radius * np.cos(theta))
    y_coords = (origins_y + radius - 1) + (radius * np.sin(-theta))

    valid_mask = draw_on_canvas(canvas, x_coords, y_coords, resolution_points, 1)

    first_valid_indices = np.argmax(valid_mask, axis=1)  # Get first valid point per layer
    last_valid_indices = x_coords.shape[1] - 1 - np.argmax(valid_mask[:, ::-1], axis=1)  # Get last valid point
    n_array = np.arange(n)

    # Get start and end points
    point_l = np.column_stack((x_coords[n_array, first_valid_indices], y_coords[n_array, first_valid_indices]))
    point_r = np.column_stack((x_coords[n_array, last_valid_indices], y_coords[n_array, last_valid_indices]))

    return point_l, point_r


def draw_line(canvas, start_points, angles=None, max_lit_px=None):
    """
    Draws multiple lines on each layer of a 3D canvas, starting from `start_points` at given `angles`.

    Args:
        canvas (np.ndarray): A 3D numpy array (n, y, x) where lines are drawn.
        start_points (np.ndarray): (n, 2) array containing (x, y) start points per layer.
        angles (np.ndarray): (n, 1) array containing angles in degrees.
        max_lit_px (int): Maximum number of pixels per line.

    Returns:
        None: Modifies `canvas` in-place.
    """
    n, y, x = canvas.shape

    if angles is None:
        angles = np.random.uniform(10, 40, size=n)

    if max_lit_px is None:
        max_lit_px = int((x * x + y * y) ** 0.5 * 3)

    angles_rad = np.radians(angles)
    dx = np.cos(angles_rad).reshape(-1, 1)
    dy = np.sin(angles_rad).reshape(-1, 1)

    # Initialize start positions
    x_coords = np.full((n, max_lit_px), start_points[:, 0].reshape(-1, 1), dtype=float)  # Shape (depth, max_lit_px)
    y_coords = np.full((n, max_lit_px), start_points[:, 1].reshape(-1, 1), dtype=float)

    # Compute all steps at once (vectorized)
    step_indices = np.arange(max_lit_px)  # Shape (max_lit_px,)
    x_coords += step_indices * dx  # Shape (depth, max_lit_px)
    x_coords = (x - 1) - abs(x_coords % (2 * (x - 1)) - (x - 1))
    y_coords += step_indices * dy

    draw_on_canvas(canvas, x_coords, y_coords, max_lit_px, 1)


select_mode_types = Literal["retain", "remove"]
def select_signal(canvas, n_points_range=(20, 30), mode:select_mode_types="retain"):
    """
    Randomly selects points from a 3D canvas to form a signal.
    Args:
        canvas (np.ndarray): A 3D numpy array (n, y, x) where the signal is selected.
        n_points_range (tuple): A tuple (min, max) specifying the range of points to select.
        mode (str): Mode to select points. "retain" retains selected points, "remove" removes selected points.

    Returns:
        np.ndarray: A 3D numpy array (n, y, x) indicating selected points.

    """
    select_mask = np.zeros_like(canvas, dtype=bool)
    n, y, x = canvas.shape

    n_points_list = np.random.randint(*n_points_range, size=n)

    c_i, c_y, c_x = np.where(canvas == 1)

    unique_i, idx_starts, counts = np.unique(c_i, return_index=True, return_counts=True)

    select_idx_mask = np.zeros(len(c_i), dtype=bool)

    for grp_idx, i_val in enumerate(unique_i):
        group_start = idx_starts[grp_idx]
        group_size = int(counts[grp_idx])
        if group_size == 0:
            continue

        n_points = int(n_points_list[i_val])
        select_count = min(n_points, group_size)

        select_idxs = np.random.choice(group_size, size=select_count, replace=False)

        select_idx_mask[group_start + select_idxs] = True

    if mode == "remove":
        select_idx_mask = ~select_idx_mask

    select_mask[c_i[select_idx_mask], c_y[select_idx_mask], c_x[select_idx_mask]] = 1

    return select_mask


# Sellmeier equation coefficients for quartz
B1, B2, B3 = 0.6961663, 0.4079426, 0.8974794
C1, C2, C3 = (0.0684043)**2, (0.1162414)**2, (9.896161)**2

def refractive_index(lambda_um):
    """
    Calculate refractive index using the Sellmeier equation.
    Args:
        lambda_um (float): Wavelength in micrometers.

    Returns:
        float: Refractive index at the given wavelength.
    """
    lambda_sq = lambda_um**2
    return np.sqrt(1 + (B1 * lambda_sq) / (lambda_sq - C1) +
                      (B2 * lambda_sq) / (lambda_sq - C2) +
                      (B3 * lambda_sq) / (lambda_sq - C3))

selected_wavelength = 0.4  # µm
n_selected = refractive_index(selected_wavelength)

# Define dispersion step size based on refractive index
quartz_step_size = n_selected * 1  # Scale for visualization

def monte_carlo_dispersion(canvas, blur_level=5, dispersion_level=1):
    """
    Simulates dispersion of photons in a 3D canvas.
    Args:
        canvas (np.ndarray): A 3D numpy array (n, y, x) where the dispersion is simulated.
        blur_level (int): Number of times to simulate dispersion.
        dispersion_level (float): Standard deviation of the dispersion.

    Returns:
        np.ndarray: A 3D numpy array (n, y, x) with dispersed photons.

    """
    new_canvas = np.zeros_like(canvas)
    n, y, x = canvas.shape

    c_i, c_y, c_x = np.where(canvas == 1)
    values = canvas[canvas != 0]

    for b in range(blur_level):
        dx = np.random.normal(0, dispersion_level, size=len(c_x))
        dy = np.random.normal(0, dispersion_level, size=len(c_y))

        new_x = np.clip(c_x + dx, 0, x-1).astype(int)
        new_y = np.clip(c_y + dy, 0, y-1).astype(int)

        new_canvas[c_i, new_y, new_x] = values

    return new_canvas

def get_continuous_time_maps(canvas, t_offset=0):
    """
    get continuous time map on a 3D canvas based on the lowest y index.
    Args:
        canvas (np.ndarray): A 3D numpy array (n, y, x) where the time values are set.
        t_offset (int): Time offset to add to the time values.

    Returns:
        np.ndarray: A 3D numpy array (n, y, x) with continuous time sequence.

    """
    n, y, x = canvas.shape

    # Find the lowest y index where the value is non-zero for each time step
    lowest_y_indices = np.min(np.argmax(canvas != 0, axis=1), axis=1)

    # Create a (n, y) time map, where each row corresponds to a different n
    y_indices = np.arange(y).reshape(1, -1 ,1) # Shape (1, y, 1)
    lowest_y_indices = lowest_y_indices.reshape(-1, 1 ,1)  # Shape (n, 1, 1)

    # Compute the time map across all y positions for all n
    time_maps = np.maximum(0, y_indices - lowest_y_indices + 1 + t_offset)

    # Expand across the x dimension
    time_maps = np.repeat(time_maps, x, axis=2)  # Shape (n, y, x)

    return time_maps

def get_random_time_maps(canvas, t_start, t_end):
    """
    get random time map on a 3D canvas based on the lowest y index.
    Args:
        canvas (np.ndarray): A 3D numpy array (n, y, x) where the time values are set.
        t_start (np.ndarray): A 3D numpy array (n, 1, 1) where the time values start.
        t_end (np.ndarray): A 3D numpy array (n, 1, 1) where the time values end.

    Returns:
        np.ndarray: A 3D numpy array (t, y, x) with random time values.

    """
    t_start = t_start.reshape(-1, 1, 1) # Shape (n, 1, 1)
    t_end = t_end.reshape(-1, 1, 1)  # Shape (n, 1, 1)

    random_times = np.random.randint(t_start, t_end, canvas.shape)

    return random_times

def generate_binary_noise(*dim, p=0.001):
    """
    Generate binary noise map.
    Args:
        *dim: Dimensions of the noise map.
        p: Probability of a pixel being set to 1.

    Returns:
        np.ndarray: A 3D numpy array (n, y, x) with binary noise map.
    """
    random_map = np.random.rand(*dim)
    return (random_map < p).astype(np.float32)

mode_types = Literal[
    "normal", 'normal_norm',
    "2channel", '2channel_norm',
    "full_construction", 'full_construction_norm'
]
class TORCHData(Dataset):
    def __init__(self, x=88, y=128, t_offset=0, num_data=1, auto_generate=True,
                 signal_count=(20, 30), signal_select_mode:select_mode_types="retain",
                 noise_density=0.01, noise_magnitude=1,
                 blur_level=1, dispersion_level=quartz_step_size, mode:mode_types="normal",
                 data_precision=torch.float32):
        self.x = x
        self.y = y
        self.t_offset = t_offset
        self.num_data = num_data
        self.signal_count = signal_count
        self.signal_select_mode = signal_select_mode
        self.noise_density = noise_density
        self.noise_magnitude = noise_magnitude
        self.blur_level = blur_level
        self.dispersion_level = dispersion_level
        self.mode = mode
        self.data_precision = data_precision
        shape = (self.num_data, self.y, self.x)
        self.original = np.zeros(shape, dtype=np.float32)
        self.original_time = np.zeros(shape, dtype=np.float32)
        self.signal = np.zeros(shape, dtype=np.float32)
        self.noise = np.zeros(shape, dtype=np.float32)
        self.noise_count = np.zeros(num_data, dtype=np.float32)
        self.sn = np.zeros(shape, dtype=np.float32)
        self.signal_time = np.zeros(shape, dtype=np.float32)
        self.noise_time = np.zeros(shape, dtype=np.float32)
        self.sn_time = np.zeros(shape, dtype=np.float32)
        self.input_data = np.zeros(shape, dtype=np.float32)
        self.target_data = np.zeros(shape, dtype=np.float32)
        self.predict_data = np.zeros(shape, dtype=np.float32)

        self.select_mask = np.zeros(shape, dtype=np.float32)
        self.continuous_time_maps = np.zeros(shape, dtype=np.float32)
        self.noise_map = np.zeros(shape, dtype=np.float32)
        self.t_start = np.zeros(num_data, dtype=np.float32)
        self.t_end = np.zeros(num_data, dtype=np.float32)
        self.random_time_maps = np.zeros(shape, dtype=np.float32)

        self.original_time_norm = np.zeros(shape, dtype=np.float32)
        self.signal_time_norm = np.zeros(shape, dtype=np.float32)
        self.noise_time_norm = np.zeros(shape, dtype=np.float32)
        self.sn_time_norm = np.zeros(shape, dtype=np.float32)

        if auto_generate:
            self.generate()
        self.set_input_target()


    def generate(self):
        pl, pr = draw_parabola(self.original)
        angles = np.random.uniform(10, 40, size=self.num_data)
        draw_line(self.original, pr, angles=angles)
        draw_line(self.original, pl, angles=angles)

        self.select_mask = select_signal(self.original, n_points_range=self.signal_count, mode=self.signal_select_mode)
        self.signal = self.original * self.select_mask

        self.signal = monte_carlo_dispersion(self.signal, blur_level=self.blur_level, dispersion_level=self.dispersion_level)

        self.continuous_time_maps = get_continuous_time_maps(self.original, self.t_offset)
        self.original_time = self.original * self.continuous_time_maps
        self.signal_time = self.signal * self.continuous_time_maps

        self.noise_map = generate_binary_noise(*(self.num_data, self.y, self.x), p=self.noise_density)
        self.noise = self.noise_map * self.noise_magnitude
        self.noise[self.signal != 0] = 0
        self.noise_count = np.where(self.noise_map, 1, 0).sum(axis=(1, 2))

        self.t_start = np.ones(self.num_data) + self.t_offset
        self.t_end = np.max(self.original_time, axis=(1, 2)) + 10
        self.random_time_maps = get_random_time_maps(self.original, self.t_start, self.t_end)
        self.noise_time = self.noise * self.random_time_maps

        self.sn = self.signal + self.noise
        self.sn_time = self.signal_time + self.noise_time

        self.original_time_norm = self.original_time / self.t_end.reshape(-1, 1, 1)
        self.signal_time_norm = self.signal_time / self.t_end.reshape(-1, 1, 1)
        self.noise_time_norm = self.noise_time / self.t_end.reshape(-1, 1, 1)
        self.sn_time_norm = self.sn_time / self.t_end.reshape(-1, 1, 1)

    def set_input_target(self):
        if self.mode == "2channel":
            self.input_data = np.stack((self.sn_time, self.sn), axis=1)
            self.target_data = np.stack((self.original_time, self.original), axis=1)
        elif self.mode == "2channel_norm":
            self.input_data = np.stack((self.sn_time_norm, self.sn), axis=1)
            self.target_data = np.stack((self.original_time_norm, self.original), axis=1)
        elif self.mode == "full_construction":
            self.input_data = self.sn_time
            self.target_data = self.original_time
        elif self.mode == "full_construction_norm":
            self.input_data = self.sn_time_norm
            self.target_data = self.original_time_norm
        elif self.mode == "normal_norm":
            self.input_data = self.sn_time_norm
            self.target_data = self.signal_time_norm
        else:
            self.input_data = self.sn_time
            self.target_data = self.signal_time
        self.input_data, self.target_data = self.torch_input_target()

    def torch_input_target(self):
        return (torch.tensor(self.input_data, dtype=self.data_precision).unsqueeze(1),
                torch.tensor(self.target_data, dtype=self.data_precision).unsqueeze(1))

    def dataloader(self, batch_size=1, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]
