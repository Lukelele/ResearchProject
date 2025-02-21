import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score, f1_score

def get_metrics(original, denoised):
    original_binary = (original > 0).float()
    denoised_binary = (denoised > 0).float()

    mse = calculate_mse_torch(original, denoised)
    psnr = calculate_psnr_torch(original, denoised)
    ssim = calculate_ssim_torch(original, denoised)
    auc = roc_auc_score(original_binary.flatten().cpu().numpy(), denoised.flatten().cpu().numpy())
    f1 = f1_score(original_binary.flatten().cpu().numpy(), denoised_binary.flatten().cpu().numpy())

    dict_metrics = {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim,
        "ROC AUC": float(auc),
        "F1": f1
    }
    return dict_metrics


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

    return ssim_map.mean().item()
