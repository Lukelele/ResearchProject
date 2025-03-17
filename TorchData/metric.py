import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score, f1_score

def get_metrics(original, denoised, noisy=None):
    original_binary = (original > 0).float()
    denoised_binary = (denoised > 0).float()
    noisy_binary = (noisy > 0).float()

    signal_retention_rate = calculate_signal_retention_rate(original_binary, denoised_binary)
    signal_retention_std = calculate_signal_retention_std(original_binary, denoised_binary)
    if noisy is None:
        noise_removal_rate = "Not Available"
        noise_removal_std = "Not Available"
    else:
        noise_removal_rate = calculate_noise_removal_rate(original_binary, denoised_binary, noisy_binary)
        noise_removal_std = calculate_noise_removal_std(original_binary, denoised_binary, noisy_binary)
    mse = calculate_mse_torch(original, denoised)
    psnr = calculate_psnr_torch(original, denoised)
    ssim = calculate_ssim_torch(original, denoised)
    auc = roc_auc_score(original_binary.flatten().cpu().numpy(), denoised.flatten().cpu().numpy())
    f1 = calculate_f1(original_binary, denoised_binary)
    f1_std = calculate_f1_std(original_binary, denoised_binary)

    dict_metrics = {
        "Signal Retention Rate": signal_retention_rate,
        "Signal Retention Standard Deviation:": signal_retention_std,
        "Noise Removal Rate": noise_removal_rate,
        "Noise Removal Standard Deviation:": noise_removal_std,
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim,
        "ROC AUC": float(auc),
        "F1": f1,
        "F1 Standard Deviation:": f1_std
    }
    return dict_metrics


def calculate_signal_retention_rate(original, denoised):

    original_flat = original.flatten().cpu()
    denoised_flat = denoised.flatten().cpu()

    # Get the indices where both tensors have a value of 1.0
    common_indices = torch.argwhere((denoised_flat == 1.0) & (original_flat == 1.0))
    num_signal_predicted = common_indices.numel()  # Use .numel() instead of len(...squeeze())

    # Get the indices where original tensor has a value of 1.0
    original_indices = torch.argwhere(original_flat == 1.0)
    num_signal = original_indices.numel()

    # Avoid division by zero
    signal_retention = num_signal_predicted / num_signal if num_signal != 0 else float('nan')
    return signal_retention

def calculate_noise_removal_rate(original, denoised, noisy):
    # Flatten and move tensors to CPU
    noisy_flat = noisy.flatten().cpu()
    original_flat = original.flatten().cpu()
    denoised_flat = denoised.flatten().cpu()

    # Count the number of signal pixels in the noisy and original tensors
    num_signal = torch.argwhere((noisy_flat == 1.0) & (original_flat == 1.0)).numel()

    # Total noise pixels is the nonzero elements in noisy minus the signal count
    num_noise = torch.nonzero(noisy_flat).numel() - num_signal

    # Count the number of signal pixels in the denoised prediction that match the original
    num_signal_predicted = torch.argwhere((denoised_flat == 1.0) & (original_flat == 1.0)).numel()

    # Total noise predicted is the nonzero elements in denoised minus the predicted signal count
    num_noise_predicted = torch.nonzero(denoised_flat).numel() - num_signal_predicted

    # Avoid division by zero
    if num_noise == 0:
        return float('nan')

    noise_removal_rate = (num_noise - num_noise_predicted) / num_noise
    return noise_removal_rate


def calculate_signal_retention_std(original, denoised):
    retention_list = [
        calculate_signal_retention_rate(original[i].unsqueeze(0), denoised[i].unsqueeze(0))
        for i in range(len(original))
    ]

    if len(retention_list) <= 1:
        return 0.0

    return torch.std(torch.tensor(retention_list), unbiased=False).item()


def calculate_noise_removal_std(original, denoised, noisy):
    removal_list = [
        calculate_noise_removal_rate(original[i].unsqueeze(0), denoised[i].unsqueeze(0), noisy[i].unsqueeze(0))
        for i in range(len(original))
    ]

    if len(removal_list) <= 1:
        return 0.0

    return torch.std(torch.tensor(removal_list), unbiased=False).item()

def calculate_f1(original, denoised):
    return f1_score(original.flatten().cpu(), denoised.flatten().cpu())

def calculate_f1_std(original, denoised):
    print(original.shape)
    f1_list = [calculate_f1(original[i], denoised[i]) for i in range(len(original))]
    if len(f1_list) <= 1:
        return 0.0
    return torch.std(torch.tensor(f1_list), unbiased=False).item()


def calculate_mse_torch(original, denoised):
    mse = torch.mean((original - denoised) ** 2)
    return mse.item()

def calculate_psnr_torch(original, denoised):
    mse = calculate_mse_torch(original, denoised)
    if mse == 0:
        return float('inf')  # No noise in the image, PSNR is infinite.
    max_pixel = torch.max(original)
    psnr = 10 * torch.log10((max_pixel ** 2) / mse)
    return psnr.item()


def calculate_ssim_torch(original, denoised, window_size=(11, 11)):
    if isinstance(window_size, int):
        window_size = (window_size, window_size)  # Convert single int to tuple

    if original.ndim == 2:
        original = original.unsqueeze(0).unsqueeze(0)
    if denoised.ndim == 2:
        denoised = denoised.unsqueeze(0).unsqueeze(0)

    def gaussian_window(window_size_x, window_size_y, sigma=1.5):
        x = torch.tensor(
            [-(i - window_size_x // 2) ** 2 / float(2 * sigma ** 2) for i in range(window_size_x)],
            dtype=torch.float32
        )
        y = torch.tensor(
            [-(j - window_size_y // 2) ** 2 / float(2 * sigma ** 2) for j in range(window_size_y)],
            dtype=torch.float32
        )
        gauss_x = torch.exp(x)
        gauss_y = torch.exp(y)
        gauss_x /= gauss_x.sum()
        gauss_y /= gauss_y.sum()

        gauss_2d = torch.outer(gauss_x, gauss_y)
        return gauss_2d.reshape(1, 1, window_size_x, window_size_y)

    window = gaussian_window(window_size[0], window_size[1]).to(original.device)
    padding = (window_size[0] // 2, window_size[1] // 2)

    mu1 = F.conv2d(original, window, padding=padding, groups=1)
    mu2 = F.conv2d(denoised, window, padding=padding, groups=1)

    sigma1_sq = F.conv2d(original * original, window, padding=padding, groups=1) - mu1 ** 2
    sigma2_sq = F.conv2d(denoised * denoised, window, padding=padding, groups=1) - mu2 ** 2
    sigma12 = F.conv2d(original * denoised, window, padding=padding, groups=1) - mu1 * mu2

    max_val = torch.max(original)
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = torch.clamp(ssim_map, min=0.0, max=1.0)  # Ensure SSIM stays in [0, 1]

    return ssim_map.mean()
