import numpy as np
import torch
import matplotlib.pyplot as plt


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


def random_remove_points(canvas, n_remove):
    none_zero_indices = canvas.nonzero()
    if none_zero_indices.shape[0] < n_remove:
        return

    if n_remove >= 1:
        random_indices = np.random.choice(none_zero_indices.shape[0], n_remove, replace=False)
    else:
        n = int(n_remove * none_zero_indices.shape[0])
        random_indices = np.random.choice(none_zero_indices.shape[0], n, replace=False)

    for i in random_indices:
        x = none_zero_indices[i][1]
        y = none_zero_indices[i][0]
        canvas[y, x] = 0



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


def generate_binary_noise(*dim, p=0.001, magnitude=1):
    random_tensor = torch.rand(*dim)
    return (random_tensor < p).float() * magnitude

def generate_noise(data, p=0.001):
    return torch.clamp(generate_binary_noise(data.shape, p=p, magnitude=1) - data, 0, 1)


def plot3d(data, ax=None):
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
    data_np = data.numpy().squeeze()
    y, x = np.nonzero(data_np)
    time_values = data_np[y, x]
    ax.scatter(y, x, time_values, c=time_values)
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Time')
    ax.set_box_aspect(None, zoom=0.85)

def plot2d(data):
    plt.imshow(data.squeeze())
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar()

class TorchData:
    def __init__(self, t_dim, x_dim, y_dim, n_remove=10, noise_p=0.1, time_mode='random'):
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
        self.generate(noise_p, time_mode)

    def generate(self, noise_p, time_mode):
        """
        Generate datasets.
        """
        pr, pl = draw_parabola(self.signal)

        angle = np.random.randint(10, 40)
        draw_line(self.signal, pr, angle)
        draw_line(self.signal, pl, angle)
        random_remove_points(self.signal, self.n_remove)

        self.signal_time, time_end = set_continuous_time(self.signal, 100)
        self.noise = generate_noise(self.signal, p=noise_p)

        if time_mode == 'random':
            self.noise_time = set_random_time(self.noise, 100, time_end+10)
        elif time_mode == 'continuous':
            self.noise_time, _ = set_continuous_time(self.noise, 100)

        self.sn_time = self.signal_time + self.noise_time